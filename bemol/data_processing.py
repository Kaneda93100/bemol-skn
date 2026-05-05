"""
    Dans ce script, on trouvera écrit tout un tas de méthodes
    permettant de prétraiter les données pour un apprentissage 
    bien éxécuté.

    On trouvera aussi des méthodes permettant de traiter les données pour pouvoir 
    tester les capacité de SkyNet.
"""

import torch 

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import bemol as bem
from bemol.rotor import Rotor

import numpy as np
import scipy as sp
from scipy import optimize

#Rotor par défaut
mex_rotor = Rotor("bemol/rotors/mexico")

PI = np.pi
tan = np.tan
sin = np.sin

wind = 15.06 ##CF mexico/rotor.yml
omega = 44.5163679 ##CF mexico/rotor.yml
pitch = -0.040143 ##CF mexico/rotor.yml

HR = bem.rotor.mexico.hubRadius
TR = bem.rotor.mexico.tipRadius
default_radius =  mex_rotor.radius[24]
print(default_radius)

def yaw(WSA,AA,R=default_radius, HR=HR, TR=TR, A0=0.35, Phi1=-PI/9, Phi2=PI):
    """
    Modèle de yaw proposée par l'IFPEN
        WSA --> WakeSkewAngle
        AA --> AzimuthAngle
        R --> radius in [HR, TR], par défaut celui de l'élément 24 de la pale de l'hélice mexico    
    """

    k1 = (1-A0) + A0*(R - HR)/(TR - HR)
    k2 = 1 - A0*(R - HR)/(TR - HR)
    eta = R/TR

    x = k1*eta*tan(WSA/2.)*sin(AA + Phi1)
    y = k2*(1-eta)*tan(WSA/2.)*sin(AA + Phi2)

    return (1 + x + y)

def PP (WSA,AA,factor=15.*np.pi/64.,R=default_radius, HR=HR, TR=TR) :
    """
    Modèle de yaw proposée par Shepper
    """
    return 1. + factor*tan(WSA/2.)*R/TR*np.sin(AA)

def Burton (AX, YAW) : 
    """
        Fonction corrective du Skew proposée par Burton.
        Va servir à prétraiter les données de Skew données à manger à SkyNet.
            
            AX --> Axial induction
            YAW --> Yaw angle
    """
    return (0.6 * AX + 1.0) * YAW

def Knudsen(axialInduction : float, Ux : float, radius = default_radius, tStep = 0.1, alphaDynamic = 0.3, tauScale = 3.0) :
    """
        Correction de Knudsen pour la dynamic inflow (Cékoitèss ??)

        J'ai mis un solver en paramètre, pour l'instant je ne sais pas si cela me sera utile (je dirai que non)
    """
    if (tStep == 0.):
        raise ValueError(f'Using a dynamic inflow model with tStep = {tStep} does not make sense!')
    
    # lower bound for wind velocity
    kappa = 1. / (tauScale * radius / max(1.0,Ux) )
    alphaDynamic = tStep*kappa*(axialInduction - alphaDynamic) + alphaDynamic
    
    return alphaDynamic


def compute_AI (solver, az:float, yaw:float, tilt = 0.0, element = [24], precone = 0.0,pitch = pitch, omega=omega, wind=wind, tStep = 0.1):
    """
        Fonction de prétraitement de l'induction axiale.
        L'idée est d'utiliser les mêmes appels que la fonction Solve de bem.py pour NingUncoupled

        Paramètres : 
            - solver : Solver BEM
            - element : éléments de la pale sur lesquels on veut calculer l'induction axiale
            - az : angle d'azimut, radian
            - yaw : Angle de yaw, radian
            - tilt : angle de tilt, radian
            - precone : angle de precone radian
            - omega : Vitesse de rotation de l'hélice, rad/s
            - wind : Vitesse du vent, m/s
            - rad : Rayon de l'élément, m (par défaut, le rayon de l'élément 24 de la pale de l'hélice mexico)

        Retourne :
            - AI_processed : Induction axiale traitée, id est induction axiale
                             après application des différentes méthodes appellées 
                             dans la fonction Solve de bem.py.    
            - Treatment : Variables calculées durant le processus. Sont utilisées dans la fonction
                          d'après, pour pouvoir sauter l'appel du correctif Yaw
    """

    AI_processed = 0.0

    solver._axial_induction = 0.0
    solver._tangential_induction = 0.0
    section = solver.rotor.sections[element[0]] 

    angle = section.twist + pitch
    chord = section.chord
    radius = section.radius
    funDrag = section.airfoil.cd
    funLift = section.airfoil.cl

    Ux, Uy = bem.tools.calculateVelocity(wind, omega, radius, az, yaw, tilt, precone)
    
    solver.update(
        Ux = Ux, Uy = Uy, 
        angle = angle, funLift = funLift, funDrag = funDrag,
        chord=chord, radius=radius,
        )


    ResEps = solver.residuals(solver.epsilon) #Pas forcément utile (et semble, apres essais, effectivement inutile)
    ResPiOvTwo = solver.residuals(PI/2)

    """ """
    ##Cette partie induit un rescaling bizarre sur les courbes 
    if ResEps * ResPiOvTwo < 0.0:
        inflowAngle = sp.optimize.brentq(solver.residuals, solver.epsilon, PI/4,)
    else: 
        residualMinusEpsilon = solver.residuals(-solver.epsilon)
        residualMinPiOvFour = solver.residuals(-PI/4)
    
        #Ce bout de code ne semble pas utile ici 

        if residualMinusEpsilon*residualMinPiOvFour < 0.0:
            # propeller break region
            inflowAngle = sp.optimize.brentq(
            solver.residuals,-PI/4 ,-solver.epsilon,
                )
        else:
            inflowAngle = sp.optimize.brentq(
                solver.residuals,PI/2 ,PI - solver.epsilon,
                )
    

    AI_processed = solver._axial_induction

    Treatment = {"rad" : radius, "Ux" : Ux, "Uy"  : Uy, "angle" : angle, "chord" : chord, "funLift" : funLift, "funDrag" : funDrag, "tStep" : tStep}

    return AI_processed, Treatment

def AIProcess(solver, **kwargs):
    """
        L'objectif de cette fonction est de traiter les données pour pouvoir 
        comparer l'éxécution via les fonctions écrites ici et celle utilisées 
        lorsque le code BEM est utilisé.

        Elle appliquera les correctifs et retournera les listes fn, ft, a et a'.
        On y va.
    """
    Wanted = ["rad","Ux", "Uy","angle", "chord", "funLift", "funDrag", "tStep"] #Dead or alive 

    for w in Wanted :
        if w not in kwargs :
            raise KeyError (f'Argument missing : {w}')

    #Velocity
    Ux = kwargs["Ux"]
    Uy = kwargs["Uy"]
    
    #Angles and chord
    angle = kwargs["angle"]

    #Radius and geometry
    chord = kwargs["chord"]
    rad = kwargs["rad"]

    #Lift and Drag (function)
    funLift = kwargs["funLift"]
    funDrag = kwargs["funDrag"]
    
    #Time variable
    tStep = kwargs["tStep"]

    #Applying dynamicInflow correction
    solver._axial_induction = solver.corrections.dynamicInflow(solver._axial_induction, Ux, rad, tStep)

    uxRelative = Ux * (1.0 - solver._axial_induction)
    uthetaRelative = Uy * (1.0 + solver._tangential_induction)
    inflowAngle = np.arctan2(uxRelative, uthetaRelative)

    attackAngle = inflowAngle - angle
    liftCoeff = funLift(attackAngle)
    dragCoeff = funDrag(attackAngle)

    normalCoeff = liftCoeff * np.cos(attackAngle) + dragCoeff * np.sin(attackAngle)
    tangentialCoeff = -liftCoeff * np.sin(attackAngle) + dragCoeff * np.cos(attackAngle)

    uRelative = np.sqrt(uxRelative**2. + uthetaRelative**2.)

    normalForce = 0.5*solver.rho*uRelative**2.*chord*normalCoeff
    tangentialForce = 0.5*solver.rho*uRelative**2.*chord*tangentialCoeff
    
    return normalForce, tangentialForce

def compute_forces(solver,Ux,Uy, x,y, index, rho = 1.191):
    
    index = index.to(torch.int32)
    chord  = solver.rotor.sections[index].chord
    angle = solver.rotor.sections[index].twist + pitch
    funLift = solver.rotor.sections[index].airfoil.cd
    funDrag = solver.rotor.sections[index].airfoil.cl

    uxRelative = Ux * (1.0 - x)
    uthetaRelative = Uy * (1.0 + y)
    inflowAngle = torch.arctan2(uxRelative, uthetaRelative)

    attackAngle = inflowAngle - angle
    liftCoeff = funLift(attackAngle)
    dragCoeff = funDrag(attackAngle)

    normalCoeff = liftCoeff * torch.cos(attackAngle) + dragCoeff * torch.sin(attackAngle)
    tangentialCoeff = -liftCoeff * torch.sin(attackAngle) + dragCoeff * torch.cos(attackAngle)

    uRelative = torch.sqrt(uxRelative**2. + uthetaRelative**2.)

    normalForce = 0.5*rho*uRelative**2.*chord*normalCoeff
    tangentialForce = 0.5*rho*uRelative**2.*chord*tangentialCoeff

    return normalForce, tangentialForce 

def compute_velocity(wind, omega, rad, azimuth, yaw, tilt, precone) : 
    """Calculate relative velocity for a given wind configuration
        (version différentiable pour l'intégrer à une procédure d'entraînement d'un NN)
    Parameters
    ----------
    wind : float
        wind speed, m/s
    omega : float
        rotation velocity, rad/s
    rad : float
        radius
    azi : float
        azimuthal angle, radians
    yaw : float
        yaw angle, radians
    tilt : float
        tilt angle, radians
    precone: float
        precone angle, radians

    """
    wind = torch.as_tensor(wind); omega = torch.as_tensor(omega)
    rad = torch.as_tensor(rad); azimuth = torch.as_tensor(azimuth)
    yaw = torch.as_tensor(yaw); tilt = torch.as_tensor(tilt); precone = torch.as_tensor(precone)
    
    Ux = wind*(
            (torch.cos(yaw)*torch.sin(tilt)*torch.cos(azimuth)+torch.sin(yaw)*torch.sin(azimuth))*torch.sin(precone)
            + torch.cos(yaw)*torch.cos(tilt)*torch.cos(precone)
        )
    Uy = wind*(
            torch.cos(tilt)*torch.sin(precone)*torch.sin(azimuth)-torch.sin(yaw)*torch.cos(azimuth)
        ) + omega*rad*torch.cos(precone)
    return Ux, Uy

def Apply_corr(x,y) :
    return x*y

def compute_inflow_aoa(solver, Ux, Uy, angle):
        uxRelative = Ux * (1.0 - solver._axial_induction)
        uthetaRelative = Uy * (1.0 + solver._tangential_induction)
        inflowAngle = np.arctan2(uxRelative, uthetaRelative)

        attackAngle = inflowAngle - angle

        return inflowAngle, attackAngle

def AI_ning_alg(solver:bem.ning.NingUncoupled, sect:int, wind:float, omega:float, pitch:float, precone:float, tilt:float, yaw:float, azimuth:float):
        
        section = solver.rotor[sect]

        Ux, Uy = bem.tools.calculateVelocity(wind = wind, omega = omega, rad = section.radius, azi = azimuth, yaw = yaw, tilt = tilt, precone = precone)
        solver._axial_induction = 0.0
        solver._tangential_induction = 0.0

        angle = section.twist + pitch
        chord = section.chord
        radius = section.radius
        funDrag = section.airfoil.cd
        funLift = section.airfoil.cl

        # update the flow state before calculating the residuals
        solver.update(
            Ux=Ux,Uy=Uy,
            angle=angle,funLift=funLift,funDrag=funDrag,
            chord=chord,radius=radius,
            )
        
        residualEpsilon = solver.residuals(solver.epsilon)
        residualPiOvTwo = solver.residuals(bem.ning.PI_HALF)

        if residualEpsilon * residualPiOvTwo < 0.0:
            inflowAngle = optimize.brentq(solver.residuals, solver.epsilon,bem.ning.PI_HALF,)
        else:
            residualMinusEpsilon = solver.residuals(-solver.epsilon)
            residualMinPiOvFour = solver.residuals(-bem.ning.PI_QUARTER)

            if residualMinusEpsilon*residualMinPiOvFour < 0.0:
                # propeller break region
                inflowAngle = optimize.brentq(
                    solver.residuals,-bem.ning.PI_QUARTER,-solver.epsilon,
                    )
            else:
                inflowAngle = optimize.brentq(
                    solver.residuals, bem.ning.PI_HALF, bem.ning.PI - solver.epsilon,
                    )
                
        wakeSkewAngle = solver.corrections.skewAngle(solver._axial_induction,yaw)
        solver._axial_induction = solver.corrections.yawModel(solver._axial_induction,wakeSkewAngle,azimuth,radius,
                                                              solver.rotor.hubRadius,solver.rotor.tipRadius)
        uxRelative = Ux * (1.0 - solver._axial_induction)
        uthetaRelative = Uy * (1.0 + solver._tangential_induction)
        inflowAngle = np.arctan2(uxRelative, uthetaRelative)
   
        return solver._axial_induction, inflowAngle