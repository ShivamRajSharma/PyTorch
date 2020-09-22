import torch 
import torch.nn as nn

class WaveNetBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        out_channels,
        kernel_size,
        dilation,
        last_out
    ):
        super(WaveNetBlock, self).__init__()
        self.padding_len = (kernel_size -1)*dilation

        self.conv_gate = nn.Conv1d(
            input_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation
        )

        self.conv_filter = nn.Conv1d(
            input_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation
        )


        self.skip = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=1
        )
        
        self.residual = nn.Conv1d(
            input_channels,
            last_out,
            kernel_size=1
        )
    
    def forward(self, x):
        causal_padded = torch.nn.functional.pad(x, (self.padding_len, 0))
        gate_ = torch.sigmoid(self.conv_gate(causal_padded))
        filter_ = torch.tanh(self.conv_filter(causal_padded))
        gate_x_filter = torch.mul(gate_, filter_)
        skip_out = self.skip(gate_x_filter)
        residual_ = torch.add(x, skip_out)
        residual_out = self.residual(residual_)
        return skip_out, residual_out


class WaveNet(nn.Module):
    def __init__(
        self,
        input_channels,
        out_channels,
        wavenet_input_channels,
        kernel_size,
        num_wavenet_blocks,
        num_classes,
        dropout
    ):
        super(WaveNet, self).__init__()
        self.inp_conv = nn.Conv1d(input_channels, out_channels, kernel_size=3, padding=1)
        self.wavenet_block = nn.Sequential(
            *[
                WaveNetBlock(
                    wavenet_input_channels,
                    wavenet_input_channels,
                    kernel_size=kernel_size,
                    dilation=2**num,
                    last_out=int(num_classes/2)
                )
                for num in range(num_wavenet_blocks)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Conv1d(int(num_classes/2), int(num_classes/2), kernel_size=1),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(int(num_classes/2), num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        residuals = []
        x = self.inp_conv(x)
        for block in self.wavenet_block:
            x, residual_out = block(x)
            residuals.append(residual_out.unsqueeze(0))
        residuals = torch.cat(residuals, dim=0)
        residual_out = torch.sum(residuals, dim=0)
        out = self.classifier(residual_out)
        return out
        

    


if __name__ == "__main__":
    inp_ = torch.randn(32, 256, 19999)
    model = WaveNet(
        input_channels=256,
        out_channels=24,
        wavenet_input_channels=24,
        kernel_size=3,
        num_wavenet_blocks=4,
        num_classes=256,
        dropout=0.4
    )
    out = model(inp_)
    print(out.shape)



