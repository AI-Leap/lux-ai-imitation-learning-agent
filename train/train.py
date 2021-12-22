import numpy as np
import json
from pathlib import Path
import os
import random
from tqdm.notebook import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def to_label(action, obs):
    strs = action.split(' ')
    unit_id = strs[1]
    if strs[0] == 'm':
        label = {'n': 0, 'w': 1, 's': 2, 'e': 3}[strs[2]]
    elif strs[0] == 'bcity':
        label = 4
    else:
        label = None

    unit_pos = (0, 0)

    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    for update in obs["updates"]:
        strs = update.split(" ")
        if strs[0] == "u" and strs[3] == unit_id:
            unit_pos = (int(strs[4]) + x_shift, int(strs[5]) + y_shift)
    return unit_id, label, unit_pos


def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True


def create_dataset_from_json(episode_dir, team_name='Toad Brigade'):
    obses = {}
    samples = []
    append = samples.append

    episodes = [path for path in Path(episode_dir).glob(
        '*.json') if 'output' not in path.name]
    for filepath in episodes:
        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load['info']['EpisodeId']
        index = np.argmax([r or 0 for r in json_load['rewards']])

        for i in range(len(json_load['steps'])-1):
            if json_load['steps'][i][index]['status'] == 'ACTIVE':
                actions = json_load['steps'][i+1][index]['action']
                obs = json_load['steps'][i][0]['observation']

                if depleted_resources(obs):
                    break

                obs['player'] = index
                obs = dict([
                    (k, v) for k, v in obs.items()
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])

                obs_id = f'{ep_id}_{i}'
                obses[obs_id] = obs

                action_map = np.zeros((5, 32, 32))
                mask = np.zeros((5, 32, 32))

                if len(actions) > 7:
                    actions = np.random.choice(actions, 7, replace=False)

                for action in actions:
                    unit_id, label, unit_pos = to_label(action, obs)
                    if label is not None:
                        action_map[label, unit_pos[0], unit_pos[1]] = 1
                        mask[:, unit_pos[0], unit_pos[1]] = 1

                mask = mask.astype('bool')
                action_map = action_map.astype('bool')
                append((obs_id, action_map, mask))

    return obses, samples


# Input for Neural Network
# Feature map size [14,32,32] and global features size [4,4,4]
def make_input(obs):
    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    global_features = np.zeros((4, 4, 4))

    b = np.zeros((13, 32, 32), dtype=np.float32)

    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]

        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])

            # Units
            team = int(strs[2])

            if team == obs['player']:
                b[0, x, y] = 1
                b[1, x, y] = (wood + coal + uranium) / 100
            else:
                b[2, x, y] = 1
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift

            if team == obs['player']:
                b[3, x, y] = 1
                b[4, x, y] = cities[city_id]
            else:
                b[5, x, y] = 1
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 6, 'coal': 7, 'uranium': 8}[r_type], x, y] = amt / 800
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            b[9 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10

    # Day/Night Cycle
    b[11, :] = (obs['step'] % 40) / 40
    # Turns
    b[12, :] = obs['step'] / 360

    return b


class LuxDataset(Dataset):
    def __init__(self, obses, samples):
        self.obses = obses
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_id, action_map, mask = self.samples[idx]
        obs = self.obses[obs_id]
        state = make_input(obs)

        return state, action_map, mask

# Neural Network for Lux AI


class BasicConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h


class LuxNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 50, 128
        self.conv0 = BasicConv2d(13, filters, (7, 7), True)
        self.blocks = nn.ModuleList(
            [BasicConv2d(filters, filters, (5, 5), True) for _ in range(layers)])
        self.fc = BasicConv2d(filters, 5, (3, 3), False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        p = self.fc(h)
        return p


def train_model(model, dataloaders_dict, optimizer, num_epochs):
    minimum_loss = 10000
    steps = 1

    for epoch in range(num_epochs):
        model.cuda()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0

            dataloader = dataloaders_dict[phase]
            for item in dataloader:
                states = item[0].cuda().float()
                actions = item[1].cuda().float()

                mask = item[2].cuda().float()
                optimizer.zero_grad()
                criterion = nn.BCEWithLogitsLoss(weight=mask)
                steps += 1

                with torch.set_grad_enabled(phase == 'train'):
                    policy = model(states)
                    loss = criterion(policy, actions)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(policy)

                if steps % 10 == 0:
                    print(
                        f'Steps {steps}/{len(dataloader.dataset)} | Loss: {epoch_loss:.7f}', flush=True)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size

            print(
                f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.7f}', flush=True)

        traced = torch.jit.trace(model.cpu(), (torch.rand(1, 13, 32, 32)))
        if epoch_loss < minimum_loss:
            loss = (epoch_loss * 10000) // 1
            traced.save(f'/root/lily/test/models/best_model_6acts_{loss}.pth')
            minimum_loss = epoch_loss

        traced.save(f'/root/lily/test/models/epoch_6acts_{epoch + 1}.pth')


if __name__ == '__main__':
    seed = 42
    seed_everything(seed)
#     model = LuxNet()
    model = torch.jit.load('/root/lily/test/models/best_model_6acts_10.0.pth')
    print(model)

    episode_dir = 'archive'
    obses, samples = create_dataset_from_json(episode_dir)
    print('obses:', len(obses), 'samples:', len(samples), flush=True)

    train, val = train_test_split(samples, test_size=0.1, random_state=3)
    batch_size = 64
    train_loader = DataLoader(
        LuxDataset(obses, train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    val_loader = DataLoader(
        LuxDataset(obses, val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )

    dataloaders_dict = {"train": train_loader, "val": val_loader}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_model(model, dataloaders_dict, optimizer, num_epochs=10)
