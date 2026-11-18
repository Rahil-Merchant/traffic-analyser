# Split report `data/dataset_261110_split`

ratios train/valid/test = (0.8, 0.1, 0.1); 147 groups assigned as whole units; 6 groups pinned to train for being larger than a split should be

| split | groups | images written | instances | Bus | Cyclist | Light Vehicle | Long Truck | Medium Truck | Minibus Taxi | Motorcycle | Pedestrian | Short Truck |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 118 | 29281 | 146189 | 1567 | 182 | 83233 | 8921 | 2768 | 22886 | 383 | 20083 | 6166 |
| valid | 18 | 3821 | 19984 | 184 | 23 | 13670 | 993 | 350 | 2038 | 47 | 1865 | 814 |
| test | 11 | 4554 | 16021 | 81 | 26 | 10985 | 511 | 248 | 1696 | 39 | 1840 | 595 |

| split | Bus % | Cyclist % | Light Vehicle % | Long Truck % | Medium Truck % | Minibus Taxi % | Motorcycle % | Pedestrian % | Short Truck % |
|---|---|---|---|---|---|---|---|---|---|
| train | 85.5 | 78.8 | 77.1 | 85.6 | 82.2 | 86.0 | 81.7 | 84.4 | 81.4 |
| valid | 10.0 | 10.0 | 12.7 | 9.5 | 10.4 | 7.7 | 10.0 | 7.8 | 10.7 |
| test | 4.4 | 11.3 | 10.2 | 4.9 | 7.4 | 6.4 | 8.3 | 7.7 | 7.9 |

## Groups per split

**train** (118): scene_093 (5671), scene_083 (2594), scene_182 (2206), scene_084 (1835), scene_085 (1824), scene_092 (1739), scene_240 (663), scene_207 (647), scene_195 (582), scene_186 (515), scene_022 (421), scene_109 (376), scene_230 (318), scene_110 (310), scene_107 (300), scene_173 (265), scene_181 (249), scene_231 (243), scene_086 (240), scene_136 (237), scene_226 (224), scene_185 (223), scene_075 (223), scene_108 (176), scene_216 (170), scene_191 (159), scene_060 (148), scene_275 (132), scene_090 (124), scene_072 (123), scene_046 (120), scene_023 (118), scene_035 (114), scene_031 (111), scene_105 (110), scene_106 (107), scene_156 (106), scene_255 (105), scene_263 (105), scene_095 (105), scene_053 (104), scene_021 (103), scene_099 (103), scene_047 (101), scene_061 (100), scene_104 (100), scene_050 (100), scene_088 (100), scene_115 (100), scene_042 (100), scene_101 (100), scene_266 (100), scene_051 (100), scene_078 (100), scene_063 (100), scene_087 (100), scene_012 (100), scene_045 (100), scene_262 (100), scene_127 (100) …

**valid** (18): scene_077 (1420), scene_102 (730), scene_183 (335), scene_273 (269), scene_111 (201), scene_118 (121), scene_024 (101), scene_038 (100), scene_041 (100), scene_269 (90), scene_126 (89), scene_171 (81), scene_091 (70), scene_149 (47), scene_279 (35), scene_135 (30), scene_116 (1), scene_013 (1)

**test** (11): scene_166 (1049), scene_193 (973), scene_079 (749), scene_208 (681), scene_103 (456), scene_169 (207), scene_113 (131), scene_025 (105), scene_059 (85), scene_014 (70), scene_064 (48)
