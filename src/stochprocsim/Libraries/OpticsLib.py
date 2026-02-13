"""
Library for all mathematical definitions of optics 

"""

import numpy as np

flipped = False

ROUND_TO = 8

## Maths ##

def overlap(psi, basis):
    """
    Checks the overlap of psi on the basis state 'basis'
    """
    return np.square(np.abs(np.transpose(np.conjugate(psi))@basis))


## 2X2 Components ##
def HWP(theta):
    theta = np.deg2rad(theta)
    return (-1j)*np.array([[np.round(np.cos(2*theta),ROUND_TO),np.round(np.sin(2*theta),ROUND_TO)],
                           [np.round(np.sin(2*theta),ROUND_TO),np.round(-1*np.cos(2*theta),ROUND_TO)]])

def QWP(theta):
    theta = np.deg2rad(theta)
    return (1/np.sqrt(2))*np.array([[np.round(1-(1j*np.cos(2*theta)),ROUND_TO),np.round(-1j*np.sin(2*theta),ROUND_TO)],
                                    [np.round(-1j*np.sin(2*theta),ROUND_TO),np.round(1+1j*np.cos(2*theta),ROUND_TO)]])

def Mirror2(phi):
    # ASSUMING PHI RELATIVE PHASE SHIFT IN RADIANS
    m = np.array([[1,0],
        [0,np.exp(1j*phi)]])
    return m 

## 4X4 Components##
if flipped == True:
    def M4(phi, phi2):
        # ASSUMING PHI RELATIVE PHASE SHIFT IN RADIANS
        m = np.array([[1,0,0,0],
            [0,np.exp(1j*phi),0,0],
            [0,0,1,0],
            [0,0,0,np.exp(1j*phi2)]])
        return m 

    def HWP_p1(theta):
        theta = np.deg2rad(theta)
        return (-1j)*np.array([[np.round(np.cos(2*theta),ROUND_TO),np.round(np.sin(2*theta),ROUND_TO),0,0],[np.round(np.sin(2*theta),ROUND_TO),np.round(-1*np.cos(2*theta),ROUND_TO),0,0],
                               [0,0,1j,0],[0,0,0,1j]])

    def HWP_p2(theta):
        theta = np.deg2rad(theta)
        return (-1j)*np.array([[1j,0,0,0],[0,1j,0,0],
                                [0,0,np.round(np.cos(2*theta),ROUND_TO),np.round(np.sin(2*theta),ROUND_TO)],[0,0,np.round(np.sin(2*theta),ROUND_TO),np.round(-1*np.cos(2*theta),ROUND_TO)]])

    def QWP_p1(theta):
        theta = np.deg2rad(theta)
        return (1/np.sqrt(2))*np.array([[np.round(1-(1j*np.cos(2*theta)),ROUND_TO),np.round(-1j*np.sin(2*theta),ROUND_TO),0,0],[np.round(-1j*np.sin(2*theta),ROUND_TO),np.round(1+1j*np.cos(2*theta),ROUND_TO),0,0],
                                        [0,0,np.sqrt(2),0],[0,0,0,np.sqrt(2)]])

    def QWP_p2(theta):    
        theta = np.deg2rad(theta)
        return (1/np.sqrt(2))*np.array([[np.sqrt(2),0,0,0],[0,np.sqrt(2),0,0],
                                          [0,0,np.round(1-(1j*np.cos(2*theta)),ROUND_TO),np.round(-1j*np.sin(2*theta),ROUND_TO)],[0,0,np.round(-1j*np.sin(2*theta),ROUND_TO),np.round(1+(1j*np.cos(2*theta)),ROUND_TO)]])

    PBS = np.array([[1,0,0,0],
                [0,0,0,1j],
                [0,0,1,0],
                [0,1j,0,0]])
else:
    ##### NTU MODES
    PBS = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1j],
            [0, 0, 1j, 0]
        ], dtype=complex)

    def M4(theta1, theta2):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.exp(1j * theta2), 0],
            [0, 0, 0, np.exp(1j * theta1)]
        ], dtype=complex)

    def HWP_p2(theta):
        theta = np.deg2rad(theta)
        c = np.cos(2 * theta)
        s = np.sin(2 * theta)
        return (-1j) * np.array([
            [c, 0, s, 0],
            [0, 1j, 0, 0],
            [s, 0, -c, 0],
            [0, 0, 0, 1j]
        ], dtype=complex)


    def HWP_p1(theta):
        theta = np.deg2rad(theta)
        c = np.cos(2 * theta)
        s = np.sin(2 * theta)
        return (-1j) * np.array([
            [1j, 0, 0, 0],
            [0, c, 0, s],
            [0, 0, 1j, 0],
            [0, s, 0, -c]
        ], dtype=complex)

    def QWP_p1(theta):
        theta = np.deg2rad(theta)
        c = np.cos(2 * theta)
        s = np.sin(2 * theta)
        return (1 / np.sqrt(2)) * np.array([
            [np.sqrt(2), 0, 0, 0],
            [0, 1 - 1j * c, 0, -1j * s],
            [0, 0, np.sqrt(2), 0],
            [0, -1j * s, 0, 1 + 1j * c]
        ], dtype=complex)


    def QWP_p2(theta):
        theta = np.deg2rad(theta)
        c = np.cos(2 * theta)
        s = np.sin(2 * theta)
        return (1 / np.sqrt(2)) * np.array([
            [1 - 1j * c, 0, -1j * s, 0],
            [0, np.sqrt(2), 0, 0],
            [-1j * s, 0, 1 + 1j * c, 0],
            [0, 0, 0, np.sqrt(2)]
        ], dtype=complex)

#####
PBS_dag = np.conjugate(np.transpose(PBS))

def getUtot(angles):
    """ 
    Returns 4x4 Unitary Matrix.
    Input: input angles: will default to 0 if not explicitly set.

    """
    QWPf12 = QWP_p2(angles['θqf2'])

    HWPf2 = HWP_p2(angles['θhf2'])

    QWPf1= QWP_p1(angles['θqf1'])

    HWPf = HWP_p1(angles['θhf1']) 

    HWP12 = HWP_p2(angles['θh2'])

    QWP12 = QWP_p2(angles['θq2'])

    HWP1 = HWP_p1(angles['θh1'])

    QWP1 = QWP_p1(angles['θq1'])
    
    QWPin12= QWP_p2(angles['θqin2'])

    HWPin2 = HWP_p2(angles['θhin2'])

    M3 = M4(angles['Φm3'], angles['Φm1'])
       
    M2 = M4(angles['Φm2'], angles['Φm2'])
    
    M1 = M4(angles['Φm1'],angles['Φm3'])

    p = angles['pipi']

    # M0 = M4(angles['Φm0'], angles['Φm0'])

    # U = HWPf2@QWPf12@HWPf@QWPf1@PBS_dag@M3@M2@M1@HWP12@QWP12@HWP1@QWP1@PBS
    U = np.exp(-1j*p)*(HWPf2@QWPf12@HWPf@QWPf1@PBS_dag@M3@M2@M1@HWP12@QWP12@HWP1@QWP1@PBS@HWPin2@QWPin12)
    
    return U
