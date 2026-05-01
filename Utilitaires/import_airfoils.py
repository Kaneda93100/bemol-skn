import xml.etree.ElementTree as ET
import os

os.makedirs("airfoils_vortex")

path = "airfoils_vortex/"
tree = ET.parse("rotor.xml")
root = tree.getroot()

AIRFOIL = root.findall(".//AIRFOIL")


for af in AIRFOIL:
    with open(path+af.attrib["name"]+".foil", "a") as file :
        file.write("# AoA [o], Cl [.], Cd [.]\n")
        PROFILE = af.findall(".//PROFILE")
        for prof in PROFILE:
            az = float(prof.attrib["angle"])
            CL = float(prof.attrib["lift"])
            CD = float(prof.attrib["drag"])
            file.write(f"{az:<12.3f}\t{CL:<12.20f}\t{CD:<12.20f}\n")
        