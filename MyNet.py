import time
import torch  # 自己手写的网络, 预测部分
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import *
import pandas
from PIL import Image
torch.manual_seed(42)

start = time.time()

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
    "juniperus_virginiana": 175
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MydataSet(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.tran = Compose([
            CenterCrop(160),   # 先裁中心区域
            ToTensor()
            # RandomAffine(degrees=15, scale=(0.9, 1.1),
            #              translate=(0.15, 0.15), fill=(1, 1, 1))
        ])

        self.dir = []
        with open("./test.csv") as f:
            f.readline()

            for i in range(8800):
                str = f.readline().strip().split(",")

                self.dir.append(str[0])

    def __len__(self):
        return len(self.dir)

    def __getitem__(self, idx):
        with Image.open(self.dir[idx]) as img:
            img = self.tran(img)
        return img


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(4),

            nn.Conv2d(32, 48, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 48, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(2),

            nn.Flatten(),
            # 不用在此处加 Softmax（如果用 CrossEntropyLoss）
            nn.Linear(20 * 20 * 48, 176),
        )

        # print(self.model)

    def forward(self, x):
        return self.model(x)


def get_data_loader():
    data_set = MydataSet()
    return DataLoader(data_set, batch_size=15, shuffle=False, pin_memory=True)


test_data = get_data_loader()

p1 = time.time()
print(p1 - start)

net = torch.load('Classify Leaves.pth',
                 weights_only=False, map_location=device)
net = net.to(device)

ans = []
# optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)

cnt = 0
with torch.no_grad():
    net.eval()
    for x in test_data:  # x: 15 x 3 x 160 x 160
        x = x.to(device)
        output = net(x)
        output = output.argmax(1)
        if cnt % 50 == 0:
            print(cnt)
        cnt += 1
        for i in output:
            ans.append(int(i))

indic = {x: y for y, x in dic.items()}


with open("sample_submission.csv", "w") as f:
    print("image,label", file=f)
    cnt = 1
    for i in ans:
        print(f"images/{18352 + cnt}.jpg,{indic[i]}", sep="", file=f)
        cnt += 1


sub = pandas.read_csv("sample_submission.csv")

print(sub["label"].shape)

endtime = time.time()

print(endtime  -start)
