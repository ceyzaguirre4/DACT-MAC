import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F

# empty if cfg.SAVE_ATTENTIONS is False
atts = {
    'question': [],
    'knowledge': [],
    'ans': [],
    'halts': []
}

def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin


class ControlUnit(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.shared_control_proj = linear(cfg.MAC.DIM, cfg.MAC.DIM)
        self.position_aware = nn.ModuleList()
        for i in range(cfg.ACT.MAX_ITER):
            self.position_aware.append(linear(cfg.MAC.DIM, cfg.MAC.DIM))    # if controlInputUnshared

        self.control_question = linear(cfg.MAC.DIM * 2, cfg.MAC.DIM)
        self.attn = linear(cfg.MAC.DIM, 1)

        self.dim = cfg.MAC.DIM

    def forward(self, context, question, controls):
        cur_step = len(controls) - 1
        control = controls[-1]

        question = torch.tanh(self.shared_control_proj(question))       # TODO: avoid repeating call
        position_aware = self.position_aware[cur_step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context

        # ++ optionally concatenate words (= context)

        # optional projection (if config.controlProj) --> stacks another linear after activation

        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        if self.cfg.SAVE_ATTENTIONS:
            if cur_step == 0:
                atts['question'].append([attn.squeeze().detach().cpu().numpy()])
            else:
                atts['question'][-1].append(attn.squeeze().detach().cpu().numpy())

        next_control = (attn * context).sum(1)

        return next_control


class ReadUnit(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.read_dropout = nn.Dropout(cfg.MAC.READ_DROPOUT)
        self.mem_proj = linear(cfg.MAC.DIM, cfg.MAC.DIM)
        self.kb_proj = linear(cfg.MAC.DIM, cfg.MAC.DIM)
        self.concat = linear(cfg.MAC.DIM * 2, cfg.MAC.DIM)
        self.concat2 = linear(cfg.MAC.DIM, cfg.MAC.DIM)
        self.attn = linear(cfg.MAC.DIM, 1)

    def forward(self, memory, know, control, masks):
        ## Step 1: knowledge base / memory interactions
        last_mem = memory[-1]
        if self.training:
            if self.cfg.MAC.MEMORY_VAR_DROPOUT:
                last_mem = last_mem * masks
            else:
                last_mem = self.read_dropout(last_mem)
        know = self.read_dropout(know.permute(0,2,1))
        proj_mem = self.mem_proj(last_mem).unsqueeze(1)
        proj_know = self.kb_proj(know)
        concat = torch.cat([
            proj_mem * proj_know,
            proj_know,
            # proj_mem        # readMemConcatProj (this also makes the know above be the projection)
        ], 2)

        # project memory interactions back to hidden dimension
        concat = self.concat2(F.elu(self.concat(concat)))       # if readMemProj ++ second projection and nonlinearity if readMemAct

        ## Step 2: compute interactions with control (if config.readCtrl)
        attn = F.elu(concat * control[-1].unsqueeze(1))

        # if readCtrlConcatInter torch.cat([interactions, concat])

        # optionally concatenate knowledge base elements

        # optional nonlinearity

        attn = self.read_dropout(attn)
        attn = self.attn(attn).squeeze(2)
        attn = F.softmax(attn, 1).unsqueeze(2)

        if self.cfg.SAVE_ATTENTIONS:
            if len(control) == 2:
                atts['knowledge'].append([attn.squeeze().detach().cpu().numpy()])
            else:
                atts['knowledge'][-1].append(attn.squeeze().detach().cpu().numpy())


        read = (attn * know).sum(1)

        return read


class WriteUnit(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if cfg.MAC.SELF_ATT:
            self.control = linear(cfg.MAC.DIM, cfg.MAC.DIM)
            self.attn = linear(cfg.MAC.DIM, 1)
            self.concat = linear(cfg.MAC.DIM * 3, cfg.MAC.DIM)
        else:
            self.concat = linear(cfg.MAC.DIM * 2, cfg.MAC.DIM)

        self.self_attention = cfg.MAC.SELF_ATT

    def forward(self, memories, retrieved, controls):
        # optionally project info if config.writeInfoProj:

        # optional info nonlinearity if writeInfoAct != 'NON'

        # compute self-attention vector based on previous controls and memories
        if self.self_attention:
            selfControl = controls[-1]
            selfControl = self.control(selfControl)
            controls_cat = torch.stack(controls[:-1], 2)
            attn = selfControl.unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            # next_mem = self.W_s(attn_mem) + self.W_p(concat)


        prev_mem = memories[-1]
        # get write unit inputs: previous memory, the new info, optionally self-attention / control
        concat = torch.cat([retrieved, prev_mem], 1)

        if self.self_attention:
            concat = torch.cat([concat, attn_mem], 1)

        # project memory back to memory dimension if config.writeMemProj
        concat = self.concat(concat)

        # optional memory nonlinearity

        # write unit gate moved to RNNWrapper

        # optional batch normalization

        next_mem = concat

        return next_mem


class MACCell(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.control = ControlUnit(cfg)
        self.read = ReadUnit(cfg)
        self.write = WriteUnit(cfg)

        self.mem_0 = nn.Parameter(torch.zeros(1, cfg.MAC.DIM))
        # control0 is most often question, other times (eg. args2.txt) its a learned parameter initialized as random normal
        if not cfg.MAC.INIT_CNTRL_AS_Q:
            self.control_0 = nn.Parameter(torch.zeros(1, cfg.MAC.DIM))

        self.cfg = cfg
        self.dim = cfg.MAC.DIM

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)
        return mask

    def init_hidden(self, b_size, question):
        if self.cfg.MAC.INIT_CNTRL_AS_Q:
            control = question
        else:
            control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)
        if self.training and self.cfg.MAC.MEMORY_VAR_DROPOUT:
            memory_mask = self.get_mask(memory, self.cfg.MAC.MEM_DROPOUT)
        else:
            memory_mask = None

        controls = [control]
        memories = [memory]

        return (controls, memories), (memory_mask)

    def forward(self, inputs, state, masks):
        words, question, img = inputs
        controls, memories = state

        control = self.control(words, question, controls)
        controls.append(control)

        read = self.read(memories, img, controls, masks)
        # if config.writeDropout < 1.0:     dropouts["write"]
        memory = self.write(memories, read, controls)
        memories.append(memory)

        return controls, memories


class OutputUnit(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.question_proj = nn.Linear(cfg.MAC.DIM, cfg.MAC.DIM)
        self.classifier_out = nn.Sequential(nn.Dropout(p=cfg.MAC.OUTPUT_DROPOUT),       # output dropout outputDropout=0.85
                                        nn.Linear(cfg.MAC.DIM * 2, cfg.MAC.DIM),
                                        nn.ELU(),
                                        nn.Dropout(p=cfg.MAC.OUTPUT_DROPOUT),       # output dropout outputDropout=0.85
                                        nn.Linear(cfg.MAC.DIM, cfg.OUTPUT.DIM))     # another linear cause outClassifierDims = [512]
        xavier_uniform_(self.classifier_out[1].weight)
        xavier_uniform_(self.classifier_out[4].weight)

    def forward(self, last_mem, question):
        question = self.question_proj(question)
        cat = torch.cat([last_mem, question], 1)
        out = self.classifier_out(cat)
        return out


class DefaultWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.controller = MACCell(cfg)
        self.classifier = OutputUnit(cfg)

        self.cfg = cfg
        if self.cfg.MAC.MEMORY_GATE:
            mult = sum([
                cfg.MAC.MEMORY_GATE_INPUT_CONT,
                cfg.MAC.MEMORY_GATE_INPUT_MEM,
                cfg.MAC.MEMORY_GATE_INPUT_Q,
                ])
            self.gate = torch.nn.Sequential()
            layer_dims = [mult * cfg.MAC.DIM] + self.cfg.MAC.MEMORY_GATE_DIMS + [1]
            for i in range(len(layer_dims) - 1):
                self.gate.add_module(f'gate_dropout_{i}', nn.Dropout(p=cfg.MAC.GATE_DROPOUT))
                self.gate.add_module(f'gate_linear_{i}', nn.Linear(layer_dims[i], layer_dims[i + 1]))
                xavier_uniform_(self.gate[-1].weight)
                self.gate[-1].bias.data.zero_()
                if i != len(layer_dims) - 2:
                    self.gate.add_module(f'gate_elu_{i}', nn.ELU())

    def forward(self, *inputs):
        state, masks = self.controller.init_hidden(inputs[1].size(0), inputs[1])

        # interfered loop
        for _ in range(1, self.cfg.ACT.MAX_ITER + 1):
            state = self.controller(inputs, state, masks)

            # memory gate
            if self.cfg.MAC.MEMORY_GATE:
                controls, memories = state
                gate_input = torch.cat([
                    controls[-1] if self.cfg.MAC.MEMORY_GATE_INPUT_CONT else torch.tensor((), device=inputs[1].device),
                    memories[-1] if self.cfg.MAC.MEMORY_GATE_INPUT_MEM else torch.tensor((), device=inputs[1].device),
                    inputs[1] if self.cfg.MAC.MEMORY_GATE_INPUT_Q else torch.tensor((), device=inputs[1].device),
                ], -1)
                gate = torch.sigmoid(self.gate(gate_input) + self.cfg.MAC.MEMORY_GATE_BIAS)
                memories[-1] = gate * memories[-2] + (1 - gate) * memories[-1]

        _, memories = state

        out = self.classifier(memories[-1], inputs[1])

        return out, torch.tensor(0, dtype=torch.float32), torch.tensor(self.cfg.ACT.MAX_ITER, dtype=torch.float32)


class DACTWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.controller = MACCell(cfg)
        self.classifier = OutputUnit(cfg)

        self.cfg = cfg
        mult = sum([
            cfg.MAC.MEMORY_GATE_INPUT_CONT,
            cfg.MAC.MEMORY_GATE_INPUT_MEM,
            cfg.MAC.MEMORY_GATE_INPUT_Q,
            ])
        self.gate = torch.nn.Sequential()
        layer_dims = [mult * cfg.MAC.DIM] + self.cfg.MAC.MEMORY_GATE_DIMS + [1]
        for i in range(len(layer_dims) - 1):
            self.gate.add_module(f'gate_dropout_{i}', nn.Dropout(p=cfg.MAC.GATE_DROPOUT))
            self.gate.add_module(f'gate_linear_{i}', nn.Linear(layer_dims[i], layer_dims[i + 1]))
            xavier_uniform_(self.gate[-1].weight)
            self.gate[-1].bias.data.zero_()
            if i != len(layer_dims) - 2:
                self.gate.add_module(f'gate_elu_{i}', nn.ELU())

    def forward(self, *inputs):
        b_size = inputs[1].size(0)
        state, masks = self.controller.init_hidden(inputs[1].size(0), inputs[1])

        # use initial control to generate accumulation tensors
        _first_control = state[0][0]
        y = _first_control.new_zeros((b_size, self.cfg.OUTPUT.DIM))
        h = _first_control.new_ones((b_size, 1))
        accumulated_h = _first_control.new_zeros((b_size, 1))
        ponder_cost = _first_control.new_full((b_size, 1), self.cfg.ACT.MIN_PENALTY)
        steps = _first_control.new_ones((b_size, 1))

        # interfered loop
        for n in range(1, self.cfg.ACT.MAX_ITER + 1):
            state = self.controller(inputs, state, masks)
            controls, memories = state

            # use last control state to calculate halt
            # 1 - gate to remain consistent with mac gate formulation.
            gate_input = torch.cat([
                controls[-1] if self.cfg.MAC.MEMORY_GATE_INPUT_CONT else torch.tensor((), device=inputs[1].device),
                memories[-1] if self.cfg.MAC.MEMORY_GATE_INPUT_MEM else torch.tensor((), device=inputs[1].device),
                inputs[1] if self.cfg.MAC.MEMORY_GATE_INPUT_Q else torch.tensor((), device=inputs[1].device),
            ], -1)
            h_n = 1 - torch.sigmoid(self.gate(gate_input) + self.cfg.MAC.MEMORY_GATE_BIAS)

            # compute y_n from last memory state
            y_n = torch.softmax(self.classifier(memories[-1], inputs[1]), dim=1)

            # accumulate h and y for timestep t
            y = torch.add(torch.mul(y_n, h), torch.mul(y, 1 - h))
            h = torch.mul(h_n, h)
            if self.cfg.SAVE_ATTENTIONS:
                if n == 1:
                    atts['ans'].append([y.squeeze().detach().cpu().numpy()])
                    atts['halts'].append([h.squeeze().detach().cpu().numpy()])
                else:
                    atts['ans'][-1].append(y.squeeze().detach().cpu().numpy())
                    atts['halts'][-1].append(h.squeeze().detach().cpu().numpy())

            # break if none in batch can change ans (adaptive computation).
            if (self.training and self.cfg.ACT.HALT_TRAIN) or (not self.training and self.cfg.ACT.HALT_TEST):
                remaining_iters = self.cfg.ACT.MAX_ITER - n
                best, runner_up = torch.topk(y, k=2, dim=1)[0].t()
                h = torch.where(best.unsqueeze(1) * ((1 - h)**remaining_iters) - runner_up.unsqueeze(1) - (h * remaining_iters) > 0,
                                h.new_zeros(1), h)

                steps = torch.where(h != 0, steps + 1, steps)

                if not torch.sum(h):
                    return y, ponder_cost, steps

            # update counter
            if self.cfg.ACT.PENALTY_TYPE == 'unit':
                ponder_cost += h
            elif self.cfg.ACT.PENALTY_TYPE == 'exp':
                accumulated_h += h
                ponder_cost += torch.exp(accumulated_h - self.cfg.ACT.EXP_PENALTY_PHASE) + 1
            elif self.cfg.ACT.PENALTY_TYPE == 'linear':
                accumulated_h += h
                ponder_cost += (accumulated_h - self.cfg.ACT.LINEAR_PENALTY_PHASE).clamp(min=0) + h
            else:
                raise Exception("Not a valid Penalty Type")

        return y, ponder_cost, steps


class ACTWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.controller = MACCell(cfg)
        self.classifier = OutputUnit(cfg)

        self.cfg = cfg
        mult = sum([
            cfg.MAC.MEMORY_GATE_INPUT_CONT,
            cfg.MAC.MEMORY_GATE_INPUT_MEM,
            cfg.MAC.MEMORY_GATE_INPUT_Q,
            ])
        self.gate = torch.nn.Sequential()
        layer_dims = [mult * cfg.MAC.DIM] + self.cfg.MAC.MEMORY_GATE_DIMS + [1]
        for i in range(len(layer_dims) - 1):
            self.gate.add_module(f'gate_dropout_{i}', nn.Dropout(p=cfg.MAC.GATE_DROPOUT))
            self.gate.add_module(f'gate_linear_{i}', nn.Linear(layer_dims[i], layer_dims[i + 1]))
            xavier_uniform_(self.gate[-1].weight)
            self.gate[-1].bias.data.zero_()
            if i != len(layer_dims) - 2:
                self.gate.add_module(f'gate_elu_{i}', nn.ELU())

    def forward(self, *inputs):
        b_size = inputs[1].size(0)
        state, masks = self.controller.init_hidden(inputs[1].size(0), inputs[1])

        # use initial control to generate accumulation tensors
        _first_control = state[0][0]
        y = _first_control.new_zeros((b_size, self.cfg.OUTPUT.DIM))
        h = _first_control.new_zeros((b_size, 1))
        r_n = _first_control.new_ones((b_size, 1))

        R = _first_control.new_ones((b_size, 1))
        ponder_cost = _first_control.new_full((b_size, 1), self.cfg.ACT.MIN_PENALTY)

        ε = _first_control.new_full((1,), self.cfg.ACT.BASELINE.EPSILON)

        steps = _first_control.new_ones((b_size, 1))

        # interfered loop
        for n in range(1, self.cfg.ACT.MAX_ITER):
            state = self.controller(inputs, state, masks)
            controls, memories = state

            # use last control state to calculate halt
            gate_input = torch.cat([
                controls[-1] if self.cfg.MAC.MEMORY_GATE_INPUT_CONT else torch.tensor((), device=inputs[1].device),
                memories[-1] if self.cfg.MAC.MEMORY_GATE_INPUT_MEM else torch.tensor((), device=inputs[1].device),
                inputs[1] if self.cfg.MAC.MEMORY_GATE_INPUT_Q else torch.tensor((), device=inputs[1].device),
            ], -1)
            h_n = torch.sigmoid(self.gate(gate_input) + self.cfg.MAC.MEMORY_GATE_BIAS)

            # compute y_n from last memory state
            y_n = self.classifier(memories[-1], inputs[1])

            # check if n >= N(t) using ∑h_n
            h += h_n
            isN = (h >= (y_n.new_ones(1) - ε))

            # compute p_n
            p_n = torch.where(isN, r_n, h_n)

            # accumulate y
            y += p_n * y_n

            # break loop if none in batch can change ans (ACT)
            # if (self.training and self.cfg.ACT.HALT_TRAIN) or (not self.training and self.cfg.ACT.HALT_TEST):
            if isN.all():
                ponder_cost += R
                return y, ponder_cost, steps

            # remainder for next timestep
            r_n = torch.where(isN, y.new_zeros(1), y.new_ones(1) - h)

            # update values of R and N
            # while not n=N(t) -> R = r_{n+1}
            R = torch.where(isN, R, r_n)

            if self.cfg.ACT.PENALTY_TYPE == 'unit':
                step_penalty = 1
            elif self.cfg.ACT.PENALTY_TYPE == 'exp':
                step_penalty = torch.exp(steps - self.cfg.ACT.EXP_PENALTY_PHASE) + 1
            elif self.cfg.ACT.PENALTY_TYPE == 'linear':
                step_penalty = (steps - self.cfg.ACT.LINEAR_PENALTY_PHASE).clamp(min=0) + 1
            else:
                raise Exception("Not a valid Penalty Type")
            ponder_cost = torch.where(isN, ponder_cost, ponder_cost + step_penalty)

            steps = torch.where(isN, steps, steps + 1)

        state = self.controller(inputs, state, masks)
        controls, memories = state

        # compute y_n from last memory state
        y_n = self.classifier(memories[-1], inputs[1])

        # compute p_n
        p_n = r_n

        # accumulate y
        y += p_n * y_n

        ponder_cost += R
        return y, ponder_cost, steps


class MACNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Dropout(p=cfg.MAC.STEM_DROPOUT),             # stem dropout stemDropout=0.82
            nn.Conv2d(cfg.MAC.KNOW_DIM, cfg.MAC.DIM, cfg.MAC.STEM_FILTER_SIZE, padding=cfg.MAC.STEM_PAD),
            nn.ELU(),
            nn.Dropout(p=cfg.MAC.STEM_DROPOUT),             # stem dropout stemDropout=0.82
            nn.Conv2d(cfg.MAC.DIM, cfg.MAC.DIM, cfg.MAC.STEM_FILTER_SIZE, padding=cfg.MAC.STEM_PAD),
            nn.ELU())
        self.question_dropout = nn.Dropout(cfg.MAC.QUESTION_DROPOUT)
        self.embed = nn.Embedding(cfg.INPUT.N_VOCAB, cfg.MAC.EMBD_DIM)
        # if bi: (bidirectional)
        hDim = int(cfg.MAC.DIM / 2)
        self.lstm = nn.LSTM(cfg.MAC.EMBD_DIM, hDim,
                        # dropout=cfg.MAC.ENC_INPUT_DROPOUT,
                        batch_first=True, bidirectional=True)

        # choose different wrappers for no-act/actSmooth/actBaseline
        if not cfg.MAC.USE_ACT:
            self.actmac = DefaultWrapper(cfg)
        elif cfg.ACT.SMOOTH:
            self.actmac = DACTWrapper(cfg)
        else:
            self.actmac = ACTWrapper(cfg)

        self.dim = cfg.MAC.DIM

        self.reset()

    def reset(self):
        # from tf implementation
        xavier_uniform_(self.embed.weight)

        xavier_uniform_(self.conv[1].weight)
        self.conv[1].bias.data.zero_()
        xavier_uniform_(self.conv[4].weight)
        self.conv[4].bias.data.zero_()

    def forward(self, image, question, question_len, dropout=0.15):
        b_size = question.size(0)

        img = self.conv(image)
        img = img.view(b_size, self.dim, -1)

        embed = self.embed(question)
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len,
                                                batch_first=True)
        lstm_out, (h, _) = self.lstm(embed)
        question = torch.cat([h[0], h[1]], -1)
        question = self.question_dropout(question)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                    batch_first=True)
        out, ponder_cost, steps = self.actmac(lstm_out, question, img)

        return out, ponder_cost, steps
