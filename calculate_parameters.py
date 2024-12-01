from models.Nets import  MobileNetCifar
from prettytable import PrettyTable


model = MobileNetCifar(num_classes=100)

M1 =   ['conv1.weight',
        'bn1.weight',
        'bn1.bias',
        'layers.0.conv1.weight',
        'layers.0.bn1.weight',
        'layers.0.bn1.bias',
        'layers.0.conv2.weight',
        'layers.0.bn2.weight',
        'layers.0.bn2.bias',
        'layers.1.conv1.weight',
        'layers.1.bn1.weight',
        'layers.1.bn1.bias',
        'layers.1.conv2.weight',
        'layers.1.bn2.weight',
        'layers.1.bn2.bias',
        'layers.2.conv1.weight',
        'layers.2.bn1.weight',
        'layers.2.bn1.bias',
        'layers.2.conv2.weight',
        'layers.2.bn2.weight',
        'layers.2.bn2.bias']

M2 =   ['layers.3.conv1.weight',
        'layers.3.bn1.weight',
        'layers.3.bn1.bias',
        'layers.3.conv2.weight',
        'layers.3.bn2.weight',
        'layers.3.bn2.bias',
        'layers.4.conv1.weight',
        'layers.4.bn1.weight',
        'layers.4.bn1.bias',
        'layers.4.conv2.weight',
        'layers.4.bn2.weight',
        'layers.4.bn2.bias']

M3 =   ['layers.5.conv1.weight',
        'layers.5.bn1.weight',
        'layers.5.bn1.bias',
        'layers.5.conv2.weight',
        'layers.5.bn2.weight',
        'layers.5.bn2.bias',
        'layers.6.conv1.weight',
        'layers.6.bn1.weight',
        'layers.6.bn1.bias',
        'layers.6.conv2.weight',
        'layers.6.bn2.weight',
        'layers.6.bn2.bias',
        'layers.7.conv1.weight',
        'layers.7.bn1.weight',
        'layers.7.bn1.bias',
        'layers.7.conv2.weight',
        'layers.7.bn2.weight',
        'layers.7.bn2.bias',
        'layers.8.conv1.weight',
        'layers.8.bn1.weight',
        'layers.8.bn1.bias',
        'layers.8.conv2.weight',
        'layers.8.bn2.weight',
        'layers.8.bn2.bias',
        'layers.9.conv1.weight',
        'layers.9.bn1.weight',
        'layers.9.bn1.bias',
        'layers.9.conv2.weight',
        'layers.9.bn2.weight',
        'layers.9.bn2.bias',
        'layers.10.conv1.weight',
        'layers.10.bn1.weight',
        'layers.10.bn1.bias',
        'layers.10.conv2.weight',
        'layers.10.bn2.weight',
        'layers.10.bn2.bias']

M4 =   ['layers.11.conv1.weight',
        'layers.11.bn1.weight',
        'layers.11.bn1.bias',
        'layers.11.conv2.weight',
        'layers.11.bn2.weight',
        'layers.11.bn2.bias',
        'layers.12.conv1.weight',
        'layers.12.bn1.weight',
        'layers.12.bn1.bias',
        'layers.12.conv2.weight',
        'layers.12.bn2.weight',
        'layers.12.bn2.bias']



for M in [M1, M2, M3, M4]:
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
                if name in M:
                        params = parameter.numel()
                        table.add_row([name, params])
                        total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params}")





