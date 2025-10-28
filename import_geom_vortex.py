import xml.etree.ElementTree as ET

files = ["rotor.xml", "simulation.xml"]

with open("blade_vortex.dat", "w") as f: 
    f.write("radius	twist	chord	airfoil\n")

    ## Extraction du rayon, du chords et des airfoils

    tree1 = ET.parse(files[0])
    root1 = tree1.getroot()

    tree2 = ET.parse(files[1])
    root2 = tree2.getroot()

    Ltsr = []; chords = []; airfoils = []; twists = []
    for elem in root1.findall(".//ELEMENT"): 
        Ltsr.append(float(elem.attrib["center"]))
        chords.append(float(elem.attrib["chord"]))
        airfoils.append(elem.attrib["airfoil"])

    for elem in root2.findall(".//ELEMENT"):
        twists.append(float(elem.attrib["twist"]))

    if len(Ltsr) != 36 or len(chords) != 36 or len(airfoils) != 36 or len(twists) != 36 :
        print("ltsr : ", len(Ltsr),"\n", "chords : " , len(chords), "airfoils : ", len(airfoils), "twists : ", len(twists))
        raise ValueError("Il y a eu un problème dans l'extraction des données (pas assez ou trop ont été extraites)")
    
    for i in range(36) : 
        f.write(f"{Ltsr[i]:<12.12f}\t{twists[i]:<12.12f}\t{chords[i]:<12.12f}\t{airfoils[i]}\n")


print(Ltsr)