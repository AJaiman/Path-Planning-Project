import glob

paths = glob.glob("./Sample-Images/*.png")

print(paths[0].split("\\")[1].split(".")[0])