import matplotlib.pyplot as plt

# MAX_GFLOPS = 13.1831
MAX_GFLOPS = 79.8715

filenames = [
    "./data/output_MMult_24x8_25.m",
    "./data/output_MMult_24x8_24.m",
    "./data/output_MMult_24x8_22.m",
    "./data/output_MMult_24x8_20.m",
    "./data/output_MMult_16x8_19.m",
    "./data/output_MMult_8x8_17.m",
    "./data/output_MMult_4x4_15R.m",
    "./data/output_MMult_1x4_8.m",
    "./data/output_MMult0.m",
]

legends = list(map(lambda s: "mm_" + s[20:-2], filenames))

colors = [
    "red",
    "green",
    "blue",
    "orange",
    "purple",
    "gray",
    "brown",
    "magenta",
][0: len(filenames)]

xss: list[list[float]] = []
yss: list[list[float]] = []

for filename in filenames:
    with open(filename, "r") as f:
        xs = []
        ys = []
        lines = f.readlines()[2:-1]
        for line in lines:
            x, y = line.split(" ")[:2]
            x = float(x)
            y = float(y)
            xs.append(x)
            ys.append(y)
        xss.append(xs)
        yss.append(ys)

for i in range(len(filenames)):
    plt.plot(xss[i], yss[i], colors[i % len(colors)], label=legends[i], marker=".")

plt.ylim(bottom=0, top=MAX_GFLOPS)
plt.legend()
plt.show()
