import torch
from torch import nn
from torchvision.transforms import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import os
from PIL import Image


torch.manual_seed(42)

print(torch.cuda.is_available())

dic = {
    "maclura_pomifera": 0,
    "ulmus_rubra": 1,
    "broussonettia_papyrifera": 2,
    "prunus_virginiana": 3,
    "acer_rubrum": 4,
    "cryptomeria_japonica": 5,
    "staphylea_trifolia": 6,
    "asimina_triloba": 7,
    "diospyros_virginiana": 8,
    "tilia_cordata": 9,
    "ulmus_pumila": 10,
    "quercus_muehlenbergii": 11,
    "juglans_cinerea": 12,
    "cercis_canadensis": 13,
    "ptelea_trifoliata": 14,
    "acer_palmatum": 15,
    "catalpa_speciosa": 16,
    "abies_concolor": 17,
    "eucommia_ulmoides": 18,
    "quercus_montana": 19,
    "koelreuteria_paniculata": 20,
    "liriodendron_tulipifera": 21,
    "styrax_japonica": 22,
    "malus_pumila": 23,
    "prunus_sargentii": 24,
    "cornus_mas": 25,
    "magnolia_virginiana": 26,
    "ostrya_virginiana": 27,
    "magnolia_acuminata": 28,
    "ilex_opaca": 29,
    "acer_negundo": 30,
    "fraxinus_nigra": 31,
    "pyrus_calleryana": 32,
    "picea_abies": 33,
    "chionanthus_virginicus": 34,
    "carpinus_caroliniana": 35,
    "zelkova_serrata": 36,
    "aesculus_pavi": 37,
    "taxodium_distichum": 38,
    "carya_tomentosa": 39,
    "picea_pungens": 40,
    "carya_glabra": 41,
    "quercus_macrocarpa": 42,
    "carya_cordiformis": 43,
    "catalpa_bignonioides": 44,
    "tsuga_canadensis": 45,
    "populus_tremuloides": 46,
    "magnolia_denudata": 47,
    "crataegus_viridis": 48,
    "populus_deltoides": 49,
    "ulmus_americana": 50,
    "pinus_bungeana": 51,
    "cornus_florida": 52,
    "pinus_densiflora": 53,
    "morus_alba": 54,
    "quercus_velutina": 55,
    "pinus_parviflora": 56,
    "salix_caroliniana": 57,
    "platanus_occidentalis": 58,
    "acer_saccharum": 59,
    "pinus_flexilis": 60,
    "gleditsia_triacanthos": 61,
    "quercus_alba": 62,
    "prunus_subhirtella": 63,
    "pseudolarix_amabilis": 64,
    "stewartia_pseudocamellia": 65,
    "quercus_stellata": 66,
    "pinus_rigida": 67,
    "salix_nigra": 68,
    "quercus_acutissima": 69,
    "pinus_virginiana": 70,
    "chamaecyparis_pisifera": 71,
    "quercus_michauxii": 72,
    "prunus_pensylvanica": 73,
    "amelanchier_canadensis": 74,
    "liquidambar_styraciflua": 75,
    "pinus_cembra": 76,
    "malus_hupehensis": 77,
    "castanea_dentata": 78,
    "magnolia_stellata": 79,
    "chionanthus_retusus": 80,
    "carya_ovata": 81,
    "quercus_marilandica": 82,
    "tilia_americana": 83,
    "cedrus_atlantica": 84,
    "ulmus_parvifolia": 85,
    "nyssa_sylvatica": 86,
    "quercus_virginiana": 87,
    "acer_saccharinum": 88,
    "magnolia_macrophylla": 89,
    "crataegus_pruinosa": 90,
    "pinus_nigra": 91,
    "abies_nordmanniana": 92,
    "pinus_taeda": 93,
    "ficus_carica": 94,
    "pinus_peucea": 95,
    "populus_grandidentata": 96,
    "acer_platanoides": 97,
    "pinus_resinosa": 98,
    "salix_matsudana": 99,
    "pinus_sylvestris": 100,
    "albizia_julibrissin": 101,
    "salix_babylonica": 102,
    "pinus_echinata": 103,
    "magnolia_tripetala": 104,
    "larix_decidua": 105,
    "pinus_strobus": 106,
    "aesculus_glabra": 107,
    "ginkgo_biloba": 108,
    "quercus_cerris": 109,
    "metasequoia_glyptostroboides": 110,
    "fagus_grandifolia": 111,
    "quercus_nigra": 112,
    "juglans_nigra": 113,
    "pinus_koraiensis": 114,
    "oxydendrum_arboreum": 115,
    "morus_rubra": 116,
    "crataegus_phaenopyrum": 117,
    "pinus_wallichiana": 118,
    "tilia_europaea": 119,
    "betula_jacqemontii": 120,
    "chamaecyparis_thyoides": 121,
    "acer_ginnala": 122,
    "acer_campestre": 123,
    "pinus_pungens": 124,
    "malus_floribunda": 125,
    "picea_orientalis": 126,
    "amelanchier_laevis": 127,
    "celtis_tenuifolia": 128,
    "gymnocladus_dioicus": 129,
    "quercus_bicolor": 130,
    "malus_coronaria": 131,
    "cercidiphyllum_japonicum": 132,
    "cedrus_libani": 133,
    "betula_nigra": 134,
    "acer_pensylvanicum": 135,
    "platanus_acerifolia": 136,
    "robinia_pseudo-acacia": 137,
    "ulmus_glabra": 138,
    "crataegus_laevigata": 139,
    "quercus_coccinea": 140,
    "prunus_serotina": 141,
    "tilia_tomentosa": 142,
    "quercus_imbricaria": 143,
    "cladrastis_lutea": 144,
    "fraxinus_pennsylvanica": 145,
    "phellodendron_amurense": 146,
    "betula_lenta": 147,
    "quercus_robur": 148,
    "aesculus_flava": 149,
    "paulownia_tomentosa": 150,
    "amelanchier_arborea": 151,
    "quercus_shumardii": 152,
    "magnolia_grandiflora": 153,
    "cornus_kousa": 154,
    "betula_alleghaniensis": 155,
    "carpinus_betulus": 156,
    "aesculus_hippocastamon": 157,
    "malus_baccata": 158,
    "acer_pseudoplatanus": 159,
    "betula_populifolia": 160,
    "prunus_yedoensis": 161,
    "halesia_tetraptera": 162,
    "quercus_palustris": 163,
    "evodia_daniellii": 164,
    "ulmus_procera": 165,
    "prunus_serrulata": 166,
    "quercus_phellos": 167,
    "cedrus_deodara": 168,
    "celtis_occidentalis": 169,
    "sassafras_albidum": 170,
    "acer_griseum": 171,
    "ailanthus_altissima": 172,
    "pinus_thunbergii": 173,
    "crataegus_crus-galli": 174,
    "juniperus_virginiana": 175,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MydataSet(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        start = time.time()
        self.tran = Compose(
            [
                CenterCrop(176),  # 先裁中心区域
                ToTensor(),
                RandomAffine(degrees=15, scale=(0.9, 1.1),
                             translate=(0.1, 0.1), fill=(1, 1, 1))
            ]
        )

        self.dir = []
        self.label = torch.zeros([18353 * 2], dtype=torch.int64, device=device)
        with open("./train.csv") as f:
            f.readline()

            for i in range(18353):
                str = f.readline().strip().split(",")
                for j in range(2):
                    self.dir.append(str[0])
                    self.label[i + j] = dic[str[1]]

        end = time.time()
        print(f"loadtime cost  {end - start}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img = Image.open(self.dir[idx])
        img = self.tran(img)
        img = img.to(device)
        return img, self.label[idx]


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        residual = x if self.right is None else self.right(x)
        out = self.left(x) + residual
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_classes=176):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), # 88
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 44
        )
        self.layer1 = self._make_layer(
            inchannel=64, outchannel=64, block_num=3, stride=1, is_shortcut=False)
        self.layer2 = self._make_layer(
            inchannel=64, outchannel=128, block_num=4, stride=2) # 22
        # self.layer3 = self._make_layer(
        #     inchannel=128, outchannel=256, block_num=6, stride=2)
        # self.layer4 = self._make_layer(
        #     inchannel=256, outchannel=512, block_num=3, stride=2)
        self.classifier = nn.Linear(128 * 2 * 2, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride, is_shortcut=True):
        if is_shortcut:
            shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        else:
            shortcut = None

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for _ in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = F.avg_pool2d(x, 11)  # 2 * 2
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_data_loader():
    data_set = MydataSet()
    return DataLoader(
        data_set, batch_size=32, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0

    net.eval()
    with torch.no_grad():
        for x, y in test_data:
            # x, y = x.to(device), y.to(device)
            outputs = net(x)

            n_correct += (outputs.argmax(1) == y).sum()
            n_total += outputs.shape[0]

    return n_correct / n_total


train_data = get_data_loader()
if os.path.isfile("Classify Leaves.pth"):
    net = torch.load(
        "Classify Leaves.pth", weights_only=False, map_location=device
    )
else:
    net = ResNet()

net = net.to(device)
lo = torch.nn.CrossEntropyLoss()
lo = lo.to(device)

# print("initial accuracy:", evaluate(train_data, net))
optimizer = torch.optim.Adam(
    net.parameters(), lr=2e-3, weight_decay=1e-4)

train_epoch = 11

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, train_epoch, 1.5e-3)

# cnt = 0
start_time = time.time()
for epoch in range(train_epoch):
    net.train()

    cost = 0
    for x, y in train_data:
        # x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = net(x)
        loss = lo(output, y)
        loss.backward()
        cost += loss.item()
        optimizer.step()
        # if cnt % 400 == 0:
        #     print(f"bacth： {cnt}")
        #     end_time = time.time()
        #     print(f"time：        {end_time - start_time:.1f}")

        # cnt += 1

    print(f"loss:  {cost}")
    scheduler.step()

    # print("epoch", epoch, f"lr:{optimizer.param_groups[0]['lr']:.5f}")
    if epoch % 3 == 0:
        print("epoch", epoch, "    accuracy:",
              f"{evaluate(train_data, net).item():.3f}")

torch.save(net, "Classify Leaves.pth")
