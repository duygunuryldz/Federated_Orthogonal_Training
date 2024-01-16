from functools import reduce
from torch import nn
from torch import optim


class MLP(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_size=400,
                 hidden_layer_num=2,
                 hidden_dropout_prob=.5,
                 input_dropout_prob=.2, args=None): 
        # Configurations.
        super().__init__()
        self.input_size = input_size
        self.input_dropout_prob = input_dropout_prob
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_size = output_size

        # Layers.
        self.layers = nn.ModuleList([
            # input
            nn.Linear(self.input_size, self.hidden_size), nn.ReLU(),
            nn.Dropout(self.input_dropout_prob),
            # hidden
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
               nn.Dropout(self.hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
               nn.Dropout(self.hidden_dropout_prob),
            # output
            nn.Linear(self.hidden_size, self.output_size)
        ])
        self.ce = nn.CrossEntropyLoss()
        self.opt = optim.SGD(self.parameters(), args.lr)

    @property
    def name(self):
        return (
            'MLP'
            '-lambda{lamda}'
            '-in{input_size}-out{output_size}'
            '-h{hidden_size}x{hidden_layer_num}'
            '-dropout_in{input_dropout_prob}_hidden{hidden_dropout_prob}'
        ).format(
            lamda=self.lamda,
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            hidden_layer_num=self.hidden_layer_num,
            input_dropout_prob=self.input_dropout_prob,
            hidden_dropout_prob=self.hidden_dropout_prob,
        )

    def forward(self, x):
        return reduce(lambda x, l: l(x), self.layers, x)
           
    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


def get_MLP(args):
    return MLP(
        28*28,
        10,
        hidden_size=400,
        hidden_layer_num=2,
        hidden_dropout_prob=0.5,
        input_dropout_prob=0.2,
        args=args
    )

