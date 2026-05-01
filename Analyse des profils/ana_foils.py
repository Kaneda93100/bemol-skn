import pathlib as p
import pandas
import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt
import os

## Récupérer les chemin vers les différents dossiers contenant les profils
path_vortex = p.Path('/home/arthur/Documents/GitHub/bemol-skn/bemol/rotors/mexico_vortex/airfoils')
path_default = p.Path('/home/arthur/Documents/GitHub/bemol-skn/bemol/rotors/mexico/airfoils')


#####################################################################################################
#                                PROFILS AF_n                                                       #
#####################################################################################################

al_met_cd_vor = {}; al_met_cl_vor = {}; 
bool_cd = False; bool_cl = False #0 --> le fichier n'a pas été rencontré, 1 --> le fichier à été rencontré
name_foil = None ; save_plot = False

for foil in path_vortex.glob('*.foil') :

    ## Récupérer les Cl et les Cd dans le data frame
    data = pandas.read_csv(foil, sep = '\s+', comment = '#', names = ['AoA [o]', 'Cl [.]', 'Cd [.]'])
    Cl_coef = data['Cl [.]']; Cd_coef = data['Cd [.]'] ; AoA_vor = data['AoA [o]']

    ## Récupérer la liste des différents Cl
    for key in list(al_met_cl_vor.keys()):
        if np.array_equal(Cl_coef, al_met_cl_vor[key]) ==  True : 
            bool_cl = True
            name_foil = foil.stem
            break
    ## Récupérer la liste des différents Cl
    if bool_cl == True : 
        print(f"\nLe profil {name_foil} a déjà été rencontré.\n")
        bool_cl = False
        name_foil = None
    else :
        al_met_cl_vor.update({foil.stem : Cl_coef})


    ## Récupérer la liste des différents Cd
    for key in list(al_met_cd_vor.keys()):
        if np.array_equal(Cd_coef, al_met_cd_vor[key]) ==  True : #
            bool_cd = True
            name_foil = foil.stem
            break
    
    ## Récupérer la liste des différents Cd
    if bool_cd == True : 
        print(f"\nLe profil {name_foil} a déjà été rencontré.\n")
        bool_cd = False
        name_foil = None
    else :
        al_met_cd_vor.update({foil.stem : Cd_coef})
    

    if save_plot == True :
        fig = plt.figure(figsize=(15,10))
        plt.suptitle(foil.stem)

        plt.subplot(1,2,1)
        plt.plot(AoA_vor, Cl_coef, label = 'Lift')
        plt.grid()
        plt.legend()
        plt.xlabel('AoA')

        plt.subplot(1,2,2)
        plt.plot(AoA_vor, Cd_coef, label = 'Drag')
        plt.grid()
        plt.legend()
        plt.xlabel('AoA')

        plt.savefig('/home/arthur/Documents/GitHub/bemol-skn/Analyse des profils/figs/AF/'+foil.stem+'.png')

        #plt.show()
        plt.close()
    else :
        continue


#####################################################################################################
#                           PROFILS CLASSIQUES                                                      #
#####################################################################################################

al_met_cl_def = {}; al_met_cd_def = {}
bool_cd = False; bool_cl = False #0 --> le fichier n'a pas été rencontré, 1 --> le fichier à été rencontré
name_foil = None ; save_plot = False

for foil in path_default.glob('*.foil') :
    data = pandas.read_csv(foil, sep = '\s+', comment = '#', names = ['AoA [o]', 'Cl [.]', 'Cd [.]'])
    Cl_coef = data['Cl [.]']; Cd_coef = data['Cd [.]'] ; AoA_def = data['AoA [o]']
    
    ## Tester les Cl
    for key in list(al_met_cl_def.keys()) :
        if np.array_equal(Cl_coef, al_met_cl_def[key], ) ==  True :
            bool_cl = True
            name_foil = foil.stem
            break
    
    if bool_cl == True : 
        print(f"\nLe profil {name_foil} a déjà été rencontré.\n")
        bool_cl = False
        name_foil = None    
    else :
        al_met_cl_def.update({foil.stem : Cl_coef})
        

    ## Tester les Cd
    for key in list(al_met_cd_def.keys()):
        if np.array_equal(Cl_coef, al_met_cl_def[key]) ==  True : #
            bool_cd = True
            name_foil = foil.stem
            break

    if bool_cd == True : 
        print(f"\nLe profil {name_foil} a déjà été rencontré.\n")
        bool_cd = False
        name_foil = None
    else :
        al_met_cd_def.update({foil.stem : Cd_coef})


    if save_plot == True : 
        fig = plt.figure(figsize=(15,10))
        plt.suptitle(foil.stem)

        plt.subplot(1,2,1)
        plt.plot(AoA_def, Cl_coef, label = 'Lift')
        plt.grid()
        plt.legend()
        plt.xlabel('AoA')

        plt.subplot(1,2,2)
        plt.plot(AoA_def, Cd_coef, label = 'Drag')
        plt.grid()
        plt.legend()
        plt.xlabel('AoA')

        plt.savefig('/home/arthur/Documents/GitHub/bemol-skn/Analyse des profils/figs/classics/'+foil.stem+'.png')

        #plt.show()
        plt.close()
    else : 
        continue
AoA_vor = AoA_vor.to_numpy()
AoA_def = AoA_def.to_numpy()



print(f"\nFoils vor cd : {list(al_met_cd_vor.keys())}\n")
print(f"\nFoils vor cl : {list(al_met_cl_vor.keys())}\n")
"""
print(f"\nFoils def cd : {list(al_met_cd_def.keys())}\n")
print(f"\nFoils def cl : {list(al_met_cl_def.keys())}\n")
"""
tol = 1e-1
## Identifier les Cd communs (ou les plus proches)/np.linalg.norm(al_met_cd_vor[key_AF].to_numpy()) 
foils_cd_shared   = {}
key_vor = None
key_default = None
for key_AF in list(al_met_cd_vor.keys()):
    min = 100.
    for key_def in list(al_met_cd_def.keys()):
        if key_def in ['Cylinder', 'Tower']:
            continue    
        if np.linalg.norm(al_met_cd_vor[key_AF].to_numpy() - al_met_cd_def[key_def].to_numpy())/np.linalg.norm(al_met_cd_def[key_def].to_numpy()) <= min :
            key_vor = key_AF
            key_default = key_def
            min = np.linalg.norm(al_met_cd_vor[key_AF].to_numpy() - al_met_cd_def[key_def].to_numpy())/np.linalg.norm(al_met_cd_def[key_def].to_numpy())
    foils_cd_shared.update({key_vor : key_default})        

"""
for key_AF in list(al_met_cd_vor.keys()):  ## Parcourir les profil AF
    for key_def in list(al_met_cd_def.keys()):
        if key_def in ['Cylinder', 'Tower']:
            continue
        if np.linalg.norm(al_met_cd_vor[key_AF].to_numpy() - al_met_cd_def[key_def].to_numpy())/np.linalg.norm(al_met_cd_def[key_def].to_numpy()) < atol :
            foils_cd_shared.update({key_AF : key}) 
"""

## Identifier les Cl communs (ou les plus proches)
foils_cl_shared   = {}
key_vor = None
key_default = None
for key_AF in list(al_met_cl_vor.keys()):
    min = 100.
    for key_def in list(al_met_cl_def.keys()):
        if key_def in ['Cylinder', 'Tower']:
            continue    
        if np.linalg.norm(al_met_cl_vor[key_AF].to_numpy() - al_met_cl_def[key_def].to_numpy())/np.linalg.norm(al_met_cl_def[key_def].to_numpy()) <= min :
            key_vor = key_AF
            key_default = key_def
            min = np.linalg.norm(al_met_cl_vor[key_AF].to_numpy() - al_met_cl_def[key_def].to_numpy())/np.linalg.norm(al_met_cl_def[key_def].to_numpy())
    foils_cl_shared.update({key_vor : key_default})    
"""
for key_AF in list(al_met_cl_vor.keys()): 
    for key_def in list(al_met_cl_def.keys()):
        if key_def in ['Cylinder', 'Tower']:
            continue
        if np.linalg.norm(al_met_cl_vor[key_AF] - al_met_cl_def[key_def])/np.linalg.norm(al_met_cl_def[key_def].to_numpy())  < atol :
            foils_cl_shared.update({key_AF : key}) 
"""



for key, value in foils_cd_shared.items() : 
    fig = plt.figure(figsize=(15,10))

    plt.subplot(1,1,1)
    plt.plot(AoA_def, al_met_cd_vor[key], label = key)
    plt.plot(AoA_def, al_met_cd_def[value], label = value)
    plt.title('CD')
    plt.grid()
    plt.legend()
    plt.xlabel('AoA')

    plt.savefig('/home/arthur/Documents/GitHub/bemol-skn/Analyse des profils/CD/'+key+'_'+value)
    plt.close()

for key, value in foils_cl_shared.items() : 
    fig = plt.figure(figsize=(15,10))

    plt.subplot(1,1,1)
    plt.plot(AoA_def, al_met_cl_vor[key], label = key)
    plt.plot(AoA_def, al_met_cl_def[value], label = value)
    plt.title('CL')
    plt.grid()
    plt.legend()
    plt.xlabel('AoA')

    plt.savefig('/home/arthur/Documents/GitHub/bemol-skn/Analyse des profils/CL/'+key+'_'+value)
    plt.close()

## Affichage final
print("#"*40 + "\n"*2)
print(f"Shared foil cd : {foils_cd_shared}, len : {len(foils_cd_shared.keys())}\n")
print(f"Shared foil cl : {foils_cl_shared}\n, len : {len(foils_cl_shared.keys())}")
print("\n"*2 + "#"*40)


print('\n'*5 + "Script éxécuté.")