### Módulo de funções utilizadas no script Contagem_de_Muons_em_Subsuperficie.ipynb
### http://localhost:8888/notebooks/TCC_Contagem_de_Muons_em_Espessuras_Rocha/Numero_de_muons_em_diferentes_densidadesrho_e_comprimentoL_Stephanie.ipynb

import numpy as np
import matplotlib.pyplot as plt
#ENERGIA MÍNIMA
from IPython.display import display, Math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def calcula_Emin_pontual(E, Eu, Op, rho, l):
    L = l*1e2 # Comprimento em metros para cm
    p = L*rho # Opacidade [g/cm2] = Comprimento [cm] * Densidade [g/cm³]
    Eminp = E[np.argmin((Op - p)**2)] - Eu #/1e3 # Valor da otimização (optimum value) E [GeV] Op e p em [g/cm²]  - Eu[GeV]
#    Eminp = Eminp_ - Eu/1e3 #subtraída a massa do muon
   # print(Math(r"$\rho$: %.2f g/cm³, $\varrho$ : %.2f g/cm², $E_{min}$ : %.2f GeV , $l$ : %.0f m" % (rho, p, Eminp, L/100.0)))
    display(Math(r" $l : %.0f m, $\rho$: %.2f g/cm³, $\varrho$ : %.2f g/cm², $E_{min}$ : %.2f GeV" % ( L/100.0, rho, p, Eminp,)))
    return Eminp, p

def calcula_Emin_continuo(rho, l):
    """   \n Seguindo pro cálculo de L em 100000 partes.")"""
    L = np.linspace(10,l,100000)

    N = len(L)
    Emin = np.zeros(N)

    for i in tqdm(range(N),desc=f"Calculando Emin para rho={rho:.3f}"): 
        p = L[i]*1e2*rho
        Emin[i] = E[np.argmin((Op - p)**2)] - Eu #/1e3

    return Emin, p


def separa_resultados(dataframe, rho_estudo):
     
    df_rho = dataframe[dataframe['Densidade (g/cm³)'] == rho_estudo].reset_index()
    return df_rho


#fluxo diferencial
def Fluxo_RB(theta,E0):
    #   Reyna(2006)/Bugaev(1998)

    c=1 #velocidade da luz
    Emass=0.10566 #GeV
    R_earth=6370 #Km
    H_atm=32 #Km
    phi=np.arccos(np.sqrt(1-(1-np.cos(theta)**2)/(1+H_atm/R_earth)**2))
    p=np.sqrt((-Emass**2+E0**2)/c)
    p0=p*np.cos(theta)
    y=np.log10(p0)
    a0=0.2455
    a1=1.288
    a2=-0.2555
    a3=0.0209
    A_B=0.00253
    FB=A_B*p0**(-a3*y**3-a2*y**2-a1*y-a0)
    FRB=np.cos(theta)**3*FB
    
#     cenith = theta_in
#     theta = cenith*np.pi/180.0 # cenith angle / redians
#     E0 = np.linspace(1e0,1e4,10000) # Muon energy / GeV

#     c = 1 # Speed of light
#     m0 = 0.1056 # Muon mass in GeV/c^2
#     p = np.sqrt((E0**2-m0**2*c**4)/c**2) # Bugaev model

#     y = np.log10(p*np.cos(theta)) # Bugaev/Reyna model
#     AB = 0.00253
#     a0 = 0.2455
#     a1 = 1.288
#     a2 = -0.2555
#     a3 = 0.0209

#     Phi_Bugaev_Reyna = AB*(p**(-(a3*y**3 + a2*y**2 + a1*y + a0)))*(np.cos(theta))**3
#     Fluxo_valores.append(Phi_Bugaev_Reyna) 
    
    return FRB

def Fluxo_RH(theta,E0):
    #  Reyna(2006)/Hebbeker(2002)

    c=1  
    Emass=0.10566 #GeV
    R_earth=6370 #Km
    H_atm=32 #Km
    phi=np.arccos(np.sqrt(1-(1-np.cos(theta)**2)/(1+H_atm/R_earth)**2))
    p=np.sqrt((-Emass**2+E0**2)/c)
    p0=p*np.cos(theta)
    y=np.log10(p0)
    h1=0.133
    h2=-2.521
    h3=-5.78
    s2=-2.11
    A_H=8.6E-5
    H=h1*(y**3-5*y**2+6*y)/2+h2*(-2*y**3+9*y**2-10*y+3)/3+h3*(y**3-3*y**2+2*y)/6+s2*(y**3-6*y**2+11*y-6)/3
    FH=A_H*10**H
    FRH=np.cos(theta)**3*FH
    return FRH

def Fluxo_T(theta,E0):
    # Tanaka(2008)

    W=0.5E2
    A_T=1.8E-3
    gamma=2.7
    DeltaE=2.6
    Ep0=E0+DeltaE
    rk=0.52
    rpi=0.78
    Bpi=90
    Bk=442
    br=0.635
    FT=A_T*W*Ep0**(-gamma)*((rpi**(-1)*Bpi*(np.cos(theta))**(-1))/(Ep0+Bpi*(np.cos(theta))**(-1))
                            +br*0.36*(rk**(-1)*Bk*(np.cos(theta))**(-1))/(Ep0+Bk*(np.cos(theta))**(-1)))
    return FT


def Fluxo_Gaisser(theta, E0):
    # Gaisser  eq 30.4 p 512 pdg20
    AG = 0.14
    BG = 0.054
    gamma = 2.7
    Epion = 115/1.1
    Ekaon = 850/1.1
    rc = 0
    flux = AG*(E0**(-gamma))*(1/(1+E0*np.cos(theta)/Epion) + BG/(1+E0*np.cos(theta)/Ekaon) + rc)
    return flux

def Fluxo_GM(theta,E0):
    #   Gaisser/MUSIC (Original)

    A_0=0.14
    r_c=1E-4
    B_G=0.054
    E_pi=115./1.1 #GeV
    E_k=850./1.1 #GeV
    c=1 #velocidade da luz
    Emass=0.10566 #GeV
    R_earth=6370.0 #Km
    H_atm=32.0 #Km
    p1=0.102573
    p2=-0.068287
    p3=0.958633
    p4=0.0407253
    p5=0.817285
    phi=np.sqrt(((np.cos(theta))**2+p1**2+p2*(np.cos(theta))**(p3)+p4*(np.cos(theta))**(p5))/(1+p1**2+p2+p4))
    #phi=np.arccos(np.sqrt(1.-((1.-np.cos(theta)**2)/((1.+H_atm/R_earth)**2))))
    DE=2.06E-3*((950./np.cos(phi))-90.0)
    #E0q=((3*E0)+7./np.cos(phi))/10
    E0q=E0
    Ep_0=E0q+DE
    A_GM=A_0*1.1*(90.*np.sqrt(np.cos(phi)+0.001)/1030.)**(4.5/((E0q)*np.cos(phi)))
    gamma=2.7
    FGM=A_GM*(E0q)**(-gamma)*((1./(1+((Ep_0*np.cos(theta))/E_pi)))+(B_G/(1+((Ep_0*np.cos(theta))/E_k)))+r_c)
    return FGM

def Fluxo_GML(theta,E0):
    #   Gaisser/MUSIC (parâmetros de Lesparre)

    A_G=0.14
    r_c=1E-4
    B_G=0.054
    E_pi=115./1.1 #GeV
    E_k=850./1.1 #GeV
    c=1 #velocidade da luz
    Emass=0.10566 #GeV
    R_earth=6370.0 #Km
    H_atm=32.0 #Km
    phi=np.arccos(np.sqrt(1.-((1.-np.cos(theta)**2)/((1.+H_atm/R_earth)**2))))
    DE=2.06E-3*((1030./np.cos(phi))-120.0)
    Ep_0=E0+DE
    A_GM=A_G*(120.*np.cos(phi)/1030.)**(1.04/((E0+DE/2.)*np.cos(phi)))
    gamma=2.7
    FGML=A_GM*E0**(-gamma)*((1./(1+((Ep_0*np.cos(theta))/E_pi)))+(B_G/(1+((Ep_0*np.cos(theta))/E_k)))+r_c)
 
    return FGML

def Fluxo_Tang(theta, E0):
    AG = 0.14
    BG = 0.054
    gamma = 2.7
    E_pi = 115/1.1
    E_k = 850/1.1
    
    rc = 1e-4
    REarth = 6370.0 # km
    Hatm = 32.0 # km
    rc = 1e-4
    cosTheta = np.sqrt(1-(1-((np.cos(theta))**2))/(1+Hatm/REarth)**2)
    DeltaE0 = 0.00206*(1030.0/cosTheta -120)
    E00 = E0 + DeltaE0
    AT = AG*(120*cosTheta/1030.0)**(1.04/((E0 + DeltaE0/2)*cosTheta))

    flux = AT*(E00**(-gamma))*(1/(1+E00*cosTheta/E_pi) + BG/(1+E00*cosTheta/E_k) + rc)
    return flux

def Fluxo_Tang_Lechmann(theta, E0):
    AG = 0.14
    BG = 0.054
    gamma = 2.7
    E_pi = 115/1.1
    E_k = 850/1.1
    
    rc = 1e-4
    x = np.cos(theta)
    p1 = 0.102573
    p2 = -0.068287
    p3 = 0.958633
    p4 = 0.0407253
    p5 = 0.817285
    
    
    cosTheta = np.sqrt((x**2 + p1**2 +p2*x**p3+p4*x**p5)/(1+p1**2+p2+p4))
    E = (3*E0 + 7/cosTheta)/10.0
    
    DeltaE0 = 0.00206*(950.0/cosTheta -90)
    E00 = E0 + DeltaE0
    
    A = 1.1*(90*np.sqrt(x + 0.001)/1030.0)**(4.5/(E0 *cosTheta))

    flux = A*AG*(E**(-gamma))*(1/(1+E00*cosTheta/E_pi) + BG/(1+E00*cosTheta/E_k) + rc)
    return flux


def int_flux(Phi, E0, Eminp, dE):
    # Estimação do fluxo integrado, a partir da contagem dos muons com energia acima da energia mínima, no fluxo definido.

    N = len(Phi)
    Int_flux = 0     #subsuperfície
    Superficie = 0   #céu aberto
 
    for i in range(N):  # Loop ajustado para percorrer os índices válidos
        if np.isnan(Phi[i]) == False:
            Superficie = Superficie + Phi[i]  # Fluxo de céu aberto
            if E0[i] >= Eminp:
                Int_flux = Int_flux + Phi[i]  # Fluxo de travessia em subsuperfície (espessura)

    Superficie = Superficie*dE
    Int_flux = Int_flux*dE

    return  Int_flux, Superficie  #em segundos

def gera_coluna_int(df_rho, E0, dE, Phi):
    i_f = []
    s_f = []
    
    for i in range(len(df_rho['Energia Mínima (GeV)'])):
        #print ('Zênite : %f grad' % cenith)
        #print ('Distância : %f m' % Lm)
        print(f"{i}/{len(df_rho['Energia Mínima (GeV)'])}")
        Emin = df_rho['Energia Mínima (GeV)'][i]
        Int_fluxo, Superficie_fluxo = int_flux(Phi, E0, Emin, dE) #números em  segundo 
        i_f.append(Int_fluxo) #para um segundo 
        s_f.append(Superficie_fluxo) #para um segundo

        # fluxo de múons (medido em múons por segundo) por 86400 (o número de segundos em um dia), 
        #obtém o número de múons esperados para serem detectados em um dia        
        ##convertendo os segundos para um dia :
        print ('Fluxo de muons a céu aberto : %f cm-2 sr-1 day-1' % (Superficie_fluxo*3600)) 

        print ('Fluxo de muons transversal : %f cm-2 sr-1 sec-1' % (Int_fluxo)) # número em um segundo
        print ('Fluxo de muons transversal : %f cm-2 sr-1 hour-1' % (Int_fluxo*3600)) # número de segundos em uma hora

        print ('Fluxo de muons transversal : %f cm-2 sr-1 day-1' % (Int_fluxo*86400)) # número de segundos em um dia.
       # print("Int_fluxo e Superficie * 86500 para converter pra dia, /365 pra converter pra ano *12/365  meses")

        print("Valores de Int_fluxo e Superficie salvos sao para um segundo. Os impressos, estão multiplicados pelo número de segundos no dia.")

    df_rho["Int_fluxo"] = i_f
    df_rho["Superficie_fluxo"] = s_f

    return df_rho
              
def fluxos(angulo1, angulo2, angulo3, angulo4, E0, Lm, df_are, df_std, df_gab):

    theta = np.radians(angulo1)
    theta1 = np.radians(angulo2)
    theta2=np.radians(angulo3)
    theta3=np.radians(angulo4)
    # Comparation of all fluxes...!
    plt.figure(figsize=(8,12))
    plt.subplot(221)
    plt.plot(E0, Fluxo_RB(theta,E0),'k',label='Reyna/Bugaev',lw=2)
    plt.plot(E0, Fluxo_RH(theta,E0),'b',label='Reyna/Hebbeker',lw=2)
    plt.plot(E0, Fluxo_T(theta,E0),'r',label='Tanaka',lw=2)
    plt.plot(E0, Fluxo_Gaisser(theta, E0),'orangered',label='F Gaisser',lw=2)
    plt.plot(E0, Fluxo_GML(theta,E0),'orange',label='Gaisser/GML',lw=2)
    plt.plot(E0, Fluxo_GM(theta,E0),'hotpink',label='Gaisser/MUSIC',lw=2)
    plt.plot(E0, Fluxo_Tang(theta, E0),'purple',label=' Gaisser/Tang',lw=2)
    plt.plot(E0, Fluxo_Tang_Lechmann(theta, E0),'darkgreen',label=' Gaisser/Tang_Lechmann',lw=2)
    
    # Adicione as linhas verticais com diferentes lineamentos
    # Certifique-se de usar os índices corretos ou nomes de colunas dos seus DataFrames
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[0], color='orange', linestyle='dotted', label='Arenito L1')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[0], color='k', linestyle='dotted', label='Padrão L1')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[0], color='r', linestyle='dotted', label='Gabro L1')


    # Adicione as linhas verticais com diferentes lineamentos
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[1], color='orange', linestyle='dashed', label='Arenito L2')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[1], color='k', linestyle='dashed', label='Padrão L2')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[1], color='r', linestyle='dashed', label='Gabro L2')



    # Adicione as linhas verticais com diferentes lineamentos
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[2], color='orange', linestyle='solid', label='Arenito L3')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[2], color='k', linestyle='solid', label='Padrão L3')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[2], color='r', linestyle='solid', label='Gabro L3')


    Emin = round(df_std['Energia Mínima (GeV)'].iloc[0], 2)
    Emin1 = round(df_std['Energia Mínima (GeV)'].iloc[1], 2)
    Emin2 = round(df_std['Energia Mínima (GeV)'].iloc[2], 2)

    # Adicione o texto com as informações desejadas
    cenith_text = f'\u03B8 = {angulo1} º'
    Lm_text = f'L1: {Lm[0]} m, Emin: {Emin} GeV '
   # Emin_text = f'Emin = {Emin}'
    Lm_text2 = f'L2: {Lm[1]} m, Emin: {Emin1} GeV '
    Lm_text3 = f'L2: {Lm[2]} m, Emin: {Emin2} GeV '

    info_text = f'{cenith_text}\n Rocha Padrão: \n{Lm_text}\n{Lm_text2}\n{Lm_text3}'
    plt.text(0.35, 0.15, info_text, fontsize=12, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))


    #plt.legend(title=r'Fluxo em $\theta=$85$^{\circ}$',loc='center right',fontsize=10)
    plt.legend(title=r'Legenda',loc='center right',fontsize=11)

    plt.ylabel(r'Fluxo Diferencial  $\Phi$ (cm$^{-2}$sr$^{-1}$s$^{-1}$GeV$^{-1}$)',fontsize=14) #Differential Flux
    plt.xlabel(r'$E_0$ (GeV)',fontsize=14)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.xlim(1E1,1E4)
    #plt.xlim(1E0,1E3)
    #plt.ylim(1E-10,1E-2)

    #plt.xlim(1E0,1E3)
    #plt.ylim(1E-10,1E-2)

    plt.subplot(222)
    plt.plot(E0, Fluxo_RB(theta1,E0),'k',label='Reyna/Bugaev',lw=2)
    plt.plot(E0, Fluxo_RH(theta1,E0),'b',label='Reyna/Hebbeker',lw=2)
    plt.plot(E0, Fluxo_T(theta1,E0),'r',label='Tanaka',lw=2)
    plt.plot(E0, Fluxo_Gaisser(theta1, E0),'orangered',label='F_Gaisser',lw=2)
    plt.plot(E0, Fluxo_GML(theta1,E0),'orange',label='Gaisser/GML',lw=2)
    plt.plot(E0, Fluxo_GM(theta1,E0),'hotpink',label='Gaisser/MUSIC',lw=2)
    plt.plot(E0,  Fluxo_Tang(theta1, E0),'purple',label=' Gaisser/Tang',lw=2)
    plt.plot(E0,  Fluxo_Tang_Lechmann(theta1, E0),'darkgreen',label=' Gaisser/Tang_Lechmann',lw=2)
    # Adicione as linhas verticais com diferentes lineamentos
    # Certifique-se de usar os índices corretos ou nomes de colunas dos seus DataFrames
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[0], color='orange', linestyle='dotted', label='Arenito L1')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[0], color='k', linestyle='dotted', label='Padrão L1')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[0], color='r', linestyle='dotted', label='Gabro L1')


    # Adicione as linhas verticais com diferentes lineamentos
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[1], color='orange', linestyle='dashed', label='Arenito L2')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[1], color='k', linestyle='dashed', label='Padrão L2')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[1], color='r', linestyle='dashed', label='Gabro L2')



    # Adicione as linhas verticais com diferentes lineamentos
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[2], color='orange', linestyle='solid', label='Arenito L3')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[2], color='k', linestyle='solid', label='Padrão L3')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[2], color='r', linestyle='solid', label='Gabro L3')


#     Emin = round(df_std['Energia Mínima (GeV)'].iloc[1], 2)
#     # Adicione o texto com as informações desejadas
#     cenith_text = f'\u03B8 = {angulo2} º'
#     Lm_text = f'L = {Lm[1]} m'
#     Emin_text = f'Emin = {Emin}'
#     info_text = f'{cenith_text}\n{Lm_text}\n{Emin_text}'
#     plt.text(0.7, 0.1, info_text, fontsize=12, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    Emin = round(df_std['Energia Mínima (GeV)'].iloc[0], 2)
    Emin1 = round(df_std['Energia Mínima (GeV)'].iloc[1], 2)
    Emin2 = round(df_std['Energia Mínima (GeV)'].iloc[2], 2)

    # Adicione o texto com as informações desejadas
    cenith_text = f'\u03B8 = {angulo2} º'
    Lm_text = f'L1: {Lm[0]} m, Emin: {Emin} GeV '
   # Emin_text = f'Emin = {Emin}'
    Lm_text2 = f'L2: {Lm[1]} m, Emin: {Emin1} GeV '
    Lm_text3 = f'L2: {Lm[2]} m, Emin: {Emin2} GeV '

    info_text = f'{cenith_text}\n Rocha Padrão: \n{Lm_text}\n{Lm_text2}\n{Lm_text3}'
    plt.text(0.35, 0.15, info_text, fontsize=12, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))

    plt.legend(title=r'Fluxos e Energias',loc='center right',fontsize=10)
    plt.ylabel(r'Fluxo Diferencial  $\Phi$ (cm$^{-2}$sr$^{-1}$s$^{-1}$GeV$^{-1}$)',fontsize=14) #Differential Flux
    plt.xlabel(r'$E_0$ (GeV)',fontsize=14)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.xlim(1E1,1E4)
    #plt.xlim(1E0,1E3)
    #plt.ylim(1E-10,1E-2))

    plt.subplot(223)
    plt.plot(E0, Fluxo_RB(theta2,E0),'k',label='Reyna/Bugaev',lw=2)
    plt.plot(E0, Fluxo_RH(theta2,E0),'b',label='Reyna/Hebbeker',lw=2)
    plt.plot(E0, Fluxo_T(theta2,E0),'r',label='Tanaka',lw=2)
    plt.plot(E0, Fluxo_Gaisser(theta2, E0),'orangered',label='F_Gaisser',lw=2)
    plt.plot(E0, Fluxo_GML(theta2,E0),'orange',label='Gaisser/GML',lw=2)
    plt.plot(E0, Fluxo_GM(theta2,E0),'hotpink',label='Gaisser/MUSIC',lw=2)
    plt.plot(E0, Fluxo_Tang(theta2, E0),'purple',label=' Gaisser/Tang',lw=2)
    plt.plot(E0, Fluxo_Tang_Lechmann(theta2, E0),'darkgreen',label=' Gaisser/Tang_Lechmann',lw=2)
    # Adicione as linhas verticais com diferentes lineamentos
    # Certifique-se de usar os índices corretos ou nomes de colunas dos seus DataFrames
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[0], color='orange', linestyle='dotted', label='L1')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[0], color='k', linestyle='dotted', label='Padrão L1')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[0], color='r', linestyle='dotted', label='Gabro  L1')


    # Adicione as linhas verticais com diferentes lineamentos
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[1], color='orange', linestyle='dashed', label='Arenito L2')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[1], color='k', linestyle='dashed', label='Padrão L2')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[1], color='r', linestyle='dashed', label='Gabro  L2')



    # Adicione as linhas verticais com diferentes lineamentos
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[2], color='orange', linestyle='solid', label='Arenito L3')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[2], color='k', linestyle='solid', label='Padrão L3')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[2], color='r', linestyle='solid', label='Gabro  L3')


#     Emin = round(df_std['Energia Mínima (GeV)'].iloc[2], 2)
#     # Adicione o texto com as informações desejadas
#     cenith_text = f'\u03B8 = {angulo3} º'
#     Lm_text = f'L = {Lm[2]} m'
#     Emin_text = f'Emin = {Emin}'
#     info_text = f'{cenith_text}\n{Lm_text}\n{Emin_text}'
#     plt.text(0.7, 0.1, info_text, fontsize=12, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))
    Emin = round(df_std['Energia Mínima (GeV)'].iloc[0], 2)
    Emin1 = round(df_std['Energia Mínima (GeV)'].iloc[1], 2)
    Emin2 = round(df_std['Energia Mínima (GeV)'].iloc[2], 2)

    # Adicione o texto com as informações desejadas
    cenith_text = f'\u03B8 = {angulo3} º'
    Lm_text = f'L1: {Lm[0]} m, Emin: {Emin} GeV '
   # Emin_text = f'Emin = {Emin}'
    Lm_text2 = f'L2: {Lm[1]} m, Emin: {Emin1} GeV '
    Lm_text3 = f'L2: {Lm[2]} m, Emin: {Emin2} GeV '

    info_text = f'{cenith_text}\n Rocha Padrão: \n{Lm_text}\n{Lm_text2}\n{Lm_text3}'
    plt.text(0.35, 0.15, info_text, fontsize=12, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))
              

    plt.legend(title=r'Fluxos e Energias',loc='center right',fontsize=10)
    plt.ylabel(r'Fluxo Diferencial  $\Phi$ (cm$^{-2}$sr$^{-1}$s$^{-1}$GeV$^{-1}$)',fontsize=14) #Differential Flux
    plt.xlabel(r'$E_0$ (GeV)',fontsize=14)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.xlim(1E1,1E4)
    #plt.xlim(1E0,1E3)
    #plt.ylim(1E-10,1E-2)
    plt.subplot(224)
    plt.plot(E0, Fluxo_RB(theta3,E0),'k',label='Reyna/Bugaev',lw=2)
    plt.plot(E0, Fluxo_RH(theta3,E0),'b',label='Reyna/Hebbeker',lw=2)
    plt.plot(E0, Fluxo_T(theta3,E0),'r',label='Tanaka',lw=2)
    plt.plot(E0, Fluxo_Gaisser(theta3, E0),'orangered',label='F_Gaisser',lw=2)
    plt.plot(E0, Fluxo_GML(theta3,E0),'orange',label='Gaisser/GML',lw=2)
    plt.plot(E0, Fluxo_GM(theta3,E0),'hotpink',label='Gaisser/MUSIC',lw=2)
    plt.plot(E0,  Fluxo_Tang(theta3, E0),'purple',label=' Gaisser/Tang',lw=2)
    plt.plot(E0,  Fluxo_Tang_Lechmann(theta3, E0),'darkgreen',label=' Gaisser/Tang_Lechmann',lw=2)
    # Adicione as linhas verticais com diferentes lineamentos
    # Certifique-se de usar os índices corretos ou nomes de colunas dos seus DataFrames
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[0], color='orange', linestyle='dotted', label='Arenito L2')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[0], color='k', linestyle='dotted', label='Padrão L2')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[0], color='r', linestyle='dotted', label='Gabro  L2')


    # Adicione as linhas verticais com diferentes lineamentos
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[1], color='orange', linestyle='dashed', label='Arenito L2')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[1], color='k', linestyle='dashed', label='Padrão L2')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[1], color='r', linestyle='dashed', label='Gabro  L2')



    # Adicione as linhas verticais com diferentes lineamentos
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[2], color='orange', linestyle='solid', label='Arenito L3')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[2], color='k', linestyle='solid', label='Padrão L3')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[2], color='r', linestyle='solid', label='Gabro  L3')


#     Emin = round(df_std['Energia Mínima (GeV)'].iloc[3], 2)
#     # Adicione o texto com as informações desejadas
#     cenith_text = f'\u03B8 = {angulo4} º'
#     Lm_text = f'L = {Lm[3]} m'
#     Emin_text = f'Emin = {Emin}'
#     info_text = f'{cenith_text}\n{Lm_text}\n{Emin_text}'
#     plt.text(0.7, 0.1, info_text, fontsize=12, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))
    Emin = round(df_std['Energia Mínima (GeV)'].iloc[0], 2)
    Emin1 = round(df_std['Energia Mínima (GeV)'].iloc[1], 2)
    Emin2 = round(df_std['Energia Mínima (GeV)'].iloc[2], 2)

    # Adicione o texto com as informações desejadas
    cenith_text = f'\u03B8 = {angulo4} º'
    Lm_text = f'L1: {Lm[0]} m, Emin: {Emin} GeV '
   # Emin_text = f'Emin = {Emin}'
    Lm_text2 = f'L2: {Lm[1]} m, Emin: {Emin1} GeV '
    Lm_text3 = f'L2: {Lm[2]} m, Emin: {Emin2} GeV '

    info_text = f'{cenith_text}\n Rocha Padrão: \n{Lm_text}\n{Lm_text2}\n{Lm_text3}'
    plt.text(0.35, 0.15, info_text, fontsize=12, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))

    plt.legend(title=r'Fluxos e Energias',loc='center right',fontsize=10)
    plt.ylabel(r'Fluxo Diferencial  $\Phi$ (cm$^{-2}$sr$^{-1}$s$^{-1}$GeV$^{-1}$)',fontsize=14) #Differential Flux
    plt.xlabel(r'$E_0$ (GeV)',fontsize=14)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.xlim(1E1,1E4)
    #plt.xlim(1E0,1E3)
    #plt.ylim(1E-10,1E-2)
   

    plt.subplots_adjust(top=0.9,bottom=0,left=0,right=2,hspace=0.25,wspace=0.15)
    plt.show()
    return

def plota_fluxo_integrado(cenith, E, Lm, Phi_Gaisser, Fluxo_GM, Fluxo_RB, Fluxo_RH, df_are,df_shale, df_std, df_lime, df_gab,loc, fluxo):
    theta = cenith * np.pi / 180.0  # Conversão do ângulo cenital para radianos

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
#     plt.loglog(E, Phi_Bugaev_Reyna, 'cyan', label='Bugaev/Reyna')
#     plt.loglog(E,Phi_Gaisser, 'purple',label='Gaisser')
#     plt.loglog(E,Phi_Tang, 'darkgreen', label='Gaisser/Tang')
#     plt.loglog(E,Phi_Tang_Lechmann, 'b', label='Gaisser/Tang2')

              
    plt.loglog(E, Phi_Gaisser, 'purple',label='Gaisser')  #lesparre
    plt.loglog(E, Fluxo_GM, 'k',label='Gaisser/Music')  #lesparre
    plt.loglog(E, Fluxo_RB, 'b', label='Bugaev/Reyna') #Lesparre  e MUYSC  Phi_Bugaev_Reyna
    plt.loglog(E, Fluxo_RH,'r',label='Reyna/Hebbeker') #Lesparre

    # Adicione as linhas verticais com diferentes lineamentos
    # Certifique-se de usar os índices corretos ou nomes de colunas dos seus DataFrames
    # Adicione as linhas verticais com rótulos personalizados incorporando o valor arredondado
    label_are = r'$E_{{\mathrm{{min}}}}$ ${{\mathrm{{arenito}}}}$: {:.2f} GeV'.format(df_are["Energia Mínima (GeV)"].iloc[loc])
    label_shale = r'$E_{{\mathrm{{min}}}}$ ${{\mathrm{{arenito}}}}$: {:.2f} GeV'.format(df_shale["Energia Mínima (GeV)"].iloc[loc])
    label_std = r'$E_{{\mathrm{{min}}}}$ ${{\mathrm{{padrão}}}}$: {:.2f} GeV'.format(df_std["Energia Mínima (GeV)"].iloc[loc])
    label_lime = r'$E_{{\mathrm{{min}}}}$ ${{\mathrm{{arenito}}}}$: {:.2f} GeV'.format(df_lime["Energia Mínima (GeV)"].iloc[loc])
    label_gab = r'$E_{{\mathrm{{min}}}}$ ${{\mathrm{{gabro}}}}$: {:.2f} GeV'.format(df_gab["Energia Mínima (GeV)"].iloc[loc])
 
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[loc], color='orange', linestyle='--', label=label_are)
    plt.axvline(df_shale['Energia Mínima (GeV)'].iloc[loc], color='b', linestyle='--', label=label_shale)
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[loc], color='k', linestyle='--', label=label_std)
    plt.axvline(df_lime['Energia Mínima (GeV)'].iloc[loc], color='g', linestyle='--', label=label_lime)
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[loc], color='r', linestyle='--', label=label_gab)

    # # Adicione as linhas verticais com diferentes lineamentos
    # plt.axvline(df_are['Energia Mínima (GeV)'].iloc[1], color='orange', linestyle='dashed', label='Eminp0_200')
    # plt.axvline(df_std['Energia Mínima (GeV)'].iloc[1], color='k', linestyle='dashed', label='Eminp3_200')
    # plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[1], color='r', linestyle='dashed', label='Eminp6_200')

    # # Adicione as linhas verticais com diferentes lineamentos
    # plt.axvline(df_are['Energia Mínima (GeV)'].iloc[2], color='orange', linestyle='dotted', label='Eminp0_300')
    # plt.axvline(df_std['Energia Mínima (GeV)'].iloc[2], color='k', linestyle='dotted', label='Eminp3_300')
    # plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[2], color='r', linestyle='dotted', label='Eminp6_300')
    # Adicione as linhas verticais com diferentes lineamentos
    #plt.axvline(df_std['Energia Mínima (GeV)'].iloc[loc], color='k', linestyle='solid', label='Padrão')
    # Adicione a região sombreada

    plt.fill_between(E, 1e-15, fluxo, where=(E >= df_std['Energia Mínima (GeV)'].iloc[loc]), color='lightblue', alpha=0.5)
              
#     # Encontre os índices onde a condição é verdadeira (região preenchida)
#     indices_fill = np.where(E >= df_std['Energia Mínima (GeV)'].iloc[loc])

#     # Defina os limites para a área preenchida (valores y correspondentes)
#     y_fill = fluxo[indices_fill]

#     # Calcule a área usando a função np.trapz
#     intensidade = np.trapz(y_fill, E[indices_fill])  #area sub a curva

       
    # Arredonde o valor de Emin para duas casas decimais
    Emin = round(df_std['Energia Mínima (GeV)'].iloc[loc], 2)
    # Adicione o texto com as informações desejadas
    cenith_text = f'\u03B8 = {cenith} º'
    Lm_text = f'L = {Lm[loc]} m'
#    Emin_text = f'Emin Rocha Padrão = {Emin} GeV'
#   info_text = f'{cenith_text}\n{Lm_text}\n{Emin_text}'
 #   intensidade_text = f'Intensidade: {intensidade:.2f} GeV'
    info_text = f'{cenith_text}\n{Lm_text}'

    plt.text(0.52, 0.1, info_text, fontsize=10, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))

    plt.xlabel("$E_{0}$ (GeV)", fontsize=15)
    plt.ylabel("Fluxo $\Phi$ (GeV$^{-1}$cm$^{-2}$sr$^{-1}$s$^{-1}$)", fontsize=15)
    plt.axis([1e0,1e4,1e-15,1e0])

    plt.legend(loc='upper right', fontsize=10)#, bbox_to_anchor=(1.5, 1))
    #plt.tight_layout()
    #plt.title("Fluxo de Muons em função da energia")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()

    plt.show()
    return 

def plota_fluxo_integrado_zoom(cenith, E, Lm, Phi_Gaisser, Fluxo_GM, Fluxo_RB, Fluxo_RH, df_are,df_shale, df_std, df_lime, df_gab,loc, fluxo):
    theta = cenith * np.pi / 180.0  # Conversão do ângulo cenital para radianos

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
              
    plt.loglog(E, Phi_Gaisser, 'purple',label='Gaisser')  #lesparre
    plt.loglog(E, Fluxo_GM, 'k',label='Gaisser/Music')  #lesparre
    plt.loglog(E, Fluxo_RB, 'b', label='Bugaev/Reyna') #Lesparre  e MUYSC  Phi_Bugaev_Reyna
    plt.loglog(E, Fluxo_RH,'r',label='Reyna/Hebbeker') #Lesparre

    label_are = r'$E_{{\mathrm{{min}}}}$ ${{\mathrm{{arenito}}}}$: {:.2f} GeV'.format(df_are["Energia Mínima (GeV)"].iloc[loc])
    label_shale = r'$E_{{\mathrm{{min}}}}$ ${{\mathrm{{arenito}}}}$: {:.2f} GeV'.format(df_shale["Energia Mínima (GeV)"].iloc[loc])
    label_std = r'$E_{{\mathrm{{min}}}}$ ${{\mathrm{{padrão}}}}$: {:.2f} GeV'.format(df_std["Energia Mínima (GeV)"].iloc[loc])
    label_lime = r'$E_{{\mathrm{{min}}}}$ ${{\mathrm{{arenito}}}}$: {:.2f} GeV'.format(df_lime["Energia Mínima (GeV)"].iloc[loc])
    label_gab = r'$E_{{\mathrm{{min}}}}$ ${{\mathrm{{gabro}}}}$: {:.2f} GeV'.format(df_gab["Energia Mínima (GeV)"].iloc[loc])
 
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[loc], color='orange', linestyle='--', label=label_are)
    plt.axvline(df_shale['Energia Mínima (GeV)'].iloc[loc], color='b', linestyle='--', label=label_shale)
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[loc], color='k', linestyle='--', label=label_std)
    plt.axvline(df_lime['Energia Mínima (GeV)'].iloc[loc], color='g', linestyle='--', label=label_lime)
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[loc], color='r', linestyle='--', label=label_gab)

    plt.fill_between(E, 1e-15, fluxo, where=(E >= df_std['Energia Mínima (GeV)'].iloc[loc]), color='lightblue', alpha=0.5)
       
    Emin = round(df_std['Energia Mínima (GeV)'].iloc[loc], 2)
    cenith_text = f'\u03B8 = {cenith} º'
    Lm_text = f'L = {Lm[loc]} m'
    info_text = f'{cenith_text}\n{Lm_text}'
    plt.text(0.52, 0.1, info_text, fontsize=11, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))
  
    plt.xlabel("$E_{0}$ (GeV)", fontsize=15)
    plt.ylabel("Fluxo $\Phi$ (GeV$^{-1}$cm$^{-2}$sr$^{-1}$s$^{-1}$)", fontsize=15)
    plt.axis([1e0,1e4,1e-15,1e0])
    plt.grid()
    plt.legend(loc='upper right', fontsize=10)#, bbox_to_anchor=(1.5, 1))
    
    ax_zoom = plt.axes([0.15, 0.17, 0.13, 0.3])  # [left, bottom, width, height] 
    ax_zoom.loglog(E, Phi_Gaisser, 'purple', label='Gaisser')
    ax_zoom.loglog(E, Fluxo_GM, 'k', label='Gaisser/Music')
    ax_zoom.loglog(E, Fluxo_RB, 'b', label='Bugaev/Reyna') #Lesparre  e MUYSC  Phi_Bugaev_Reyna
    ax_zoom.loglog(E, Fluxo_RH,'r',label='Reyna/Hebbeker') #Lesparre
    ax_zoom.axvline(df_are['Energia Mínima (GeV)'].iloc[loc], color='orange', linestyle='--', label=label_are)
    ax_zoom.axvline(df_shale['Energia Mínima (GeV)'].iloc[loc], color='b', linestyle='--', label=label_shale)
    ax_zoom.axvline(df_std['Energia Mínima (GeV)'].iloc[loc], color='k', linestyle='--', label=label_std)
    ax_zoom.axvline(df_lime['Energia Mínima (GeV)'].iloc[loc], color='g', linestyle='--', label=label_lime)
    ax_zoom.axvline(df_gab['Energia Mínima (GeV)'].iloc[loc], color='r', linestyle='--', label=label_gab)

    ax_zoom.set_xlim([1e2, 7e2])  # Limites do eixo x na área de zoom
    ax_zoom.set_ylim([1e2, 1e3])  # Limites do eixo y na área de zoom
    ax_zoom.tick_params(axis='y', labelsize=8)  # Escolha o tamanho de fonte desejado
    ax_zoom.tick_params(axis='x', labelsize=6)  # Escolha o tamanho de fonte desejado

    ax_zoom.grid(which='both', axis='both')
    plt.grid(which='both', axis='both')
    plt.axis([5500, 1e4, 1e-14, 1e-12])  # xmin, xmax, ymin, ymax
    plt.grid(which='both', axis='both')
    plt.show()
    return

              
def plota_fluxo_integrado_profundidades(E, cenith,Phi_Bugaev_Reyna, Phi_Gaisser, Phi_Tang, Phi_Tang_Lechmann, df_are, df_std, df_gab):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.loglog(E, Phi_Bugaev_Reyna, 'cyan', label='Bugaev/Reyna')
    plt.loglog(E,Phi_Gaisser, 'purple',label='Gaisser')
    plt.loglog(E,Phi_Tang, 'darkgreen', label='Gaisser/Tang')
    plt.loglog(E,Phi_Tang_Lechmann, 'b', label='Gaisser/Tang2')

    # Adicione as linhas verticais com diferentes lineamentos
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[0], color='orange', linestyle='solid', label='Arenito 100m')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[0], color='k', linestyle='solid', label='Padrão 100m')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[0], color='r', linestyle='solid', label='Gabro  100m')

    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[1], color='orange', linestyle='dashed', label='Arenito 200m')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[1], color='k', linestyle='dashed', label='Padrão 200m')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[1], color='r', linestyle='dashed', label='Gabro  200m')

    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[2], color='orange', linestyle='dotted', label='Arenito 300m')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[2], color='k', linestyle='dotted', label='Padrão 300m')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[2], color='r', linestyle='dotted', label='Gabro  300m')

    Emin = round(df_std['Energia Mínima (GeV)'].iloc[0], 2)
    # Adicione o texto com as informações desejadas
    cenith_text = f'\u03B8 = {cenith} º'
    Lm_text = f'L = {Lm[0]} m'
    Emin_text = f'Emin = {Emin}'
    info_text = f'{cenith_text}\n{Lm_text}\n{Emin_text}'
    plt.text(0.7, 0.1, info_text, fontsize=12, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))

    plt.xlabel("$E_0$ (GeV)", fontsize=15)
    plt.ylabel("Fluxo $\Phi$ (GeV$^{-1}$cm$^{-2}$sr$^{-1}$s$^{-1}$)", fontsize=15)
    plt.axis([10,1e4,1e-15,1e0])

    plt.legend(loc='upper right', fontsize=15, bbox_to_anchor=(1.5, 1))
    #plt.tight_layout()
    #plt.title("Fluxo de Muons em função da energia")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()

    plt.show()
    return

def plota_fluxo_integrado_angulos(E,fluxo1, fluxo2, fluxo3, df_are, df_std, df_gab):

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.loglog(E, fluxo1, 'b', label='fluxo1')
    plt.loglog(E, fluxo2, 'purple', label='fluxo2')
    plt.loglog(E, fluxo3, 'green', label='fluxo3')

    # Adicione as linhas verticais com diferentes lineamentos
    # Certifique-se de usar os índices corretos ou nomes de colunas dos seus DataFrames
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[0], color='orange', linestyle='solid', label='Arenito 100m')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[0], color='k', linestyle='solid', label='Padrão 100m')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[0], color='r', linestyle='solid', label='Gabro  100m')


    # Adicione as linhas verticais com diferentes lineamentos
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[1], color='orange', linestyle='dashed', label='Arenito 200m')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[1], color='k', linestyle='dashed', label='Padrão 200m')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[1], color='r', linestyle='dashed', label='Gabro  200m')



    # Adicione as linhas verticais com diferentes lineamentos
    plt.axvline(df_are['Energia Mínima (GeV)'].iloc[2], color='orange', linestyle='dotted', label='Arenito 300m')
    plt.axvline(df_std['Energia Mínima (GeV)'].iloc[2], color='k', linestyle='dotted', label='Padrão 300m')
    plt.axvline(df_gab['Energia Mínima (GeV)'].iloc[2], color='r', linestyle='dotted', label='Gabro  300m')


    Emin = round(df_std['Energia Mínima (GeV)'].iloc[0], 2)
    # Adicione o texto com as informações desejadas
    cenith_text = f'\u03B8 = {cenith} º'
    Lm_text = f'L = {Lm[0]} m'
    Emin_text = f'Emin = {Emin}'
    info_text = f'{cenith_text}\n{Lm_text}\n{Emin_text}'
    plt.text(0.7, 0.1, info_text, fontsize=12, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))


    plt.xlabel("$E_0$ (GeV)", fontsize=15)
    plt.ylabel("Fluxo $\Phi$ (GeV$^{-1}$cm$^{-2}$sr$^{-1}$s$^{-1}$)", fontsize=15)
    plt.axis([10,1e4,1e-15,1e0])
    #jjustar o y, verificar os artigos qual o range que usam
    plt.legend(loc='upper right', fontsize=15, bbox_to_anchor=(1.5, 1))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()

    plt.show()
    return

#TEMPO

def tempo(df):
    #numero_muons_sem_detector = tempo * Int_fluxo
    #tempo:
    #segundo = 1
    minuto = 60 #* segundo
    hora = 60 * minuto
    dia = 24 * hora
    semana = 7 * dia
    #'Energia Mínima (GeV)', 'Opacidade (p)'
    df["Int_fluxo_hora"] = df['Int_fluxo']*hora
    df["Int_fluxo_dia"] = df['Int_fluxo']*dia                #df_std["Int_fluxo_hora"]*24
    df["Int_fluxo_semana"] = df['Int_fluxo']*semana          #df_std["Int_fluxo_dia"]*7
    df["Int_fluxo_mes"] = df['Int_fluxo']*4*semana           #df_std["Int_fluxo_semana"] *4
    df["Int_fluxo_bimestre"] = df['Int_fluxo']*2*4*semana    #df_std["Int_fluxo_mes"] *4*2
    df["Int_fluxo_trimestre"] = df['Int_fluxo']*3*4*semana   #df_std["Int_fluxo_mes"] *4*3
    df["Int_fluxo_semestre"] = df['Int_fluxo']*6*4*semana    #df_std["Int_fluxo_mes"] *4*6
    df["Int_fluxo_ano"] = df['Int_fluxo']*12*4*semana        #df_std["Int_fluxo_mes"] *4*12
    return df

#tratamento dos dados
def normaliza_dado(df_rho,df_padrao,coluna, nova_coluna0,nova_coluna1):

    # (medido - padrão) / padrão Z-score
    df_rho[nova_coluna0] = df_rho[[coluna]] / df_padrao[[coluna]]
    df_rho[nova_coluna1] = (df_rho[[coluna]] - df_padrao[[coluna]])/ df_padrao[[coluna]]
    return df_rho

def normaliza_int_flux_df(df_rho,df_padrao):
    #todos de Int_flux

    coluna = 'Int_fluxo_hora'
    nova_coluna0 = 'Hora Normalizado'
    nova_coluna1 = 'Hora Normalizado Z'
    df_rho = normaliza_dado(df_rho, df_padrao,coluna, nova_coluna0,nova_coluna1)

    coluna = 'Int_fluxo_dia'
    nova_coluna0 = 'Dia Normalizado'
    nova_coluna1 = 'Dia Normalizado Z'
    df_rho = normaliza_dado(df_rho, df_padrao, coluna, nova_coluna0,nova_coluna1)

    coluna = 'Int_fluxo_semana'
    nova_coluna0 = 'Semana Normalizado'
    nova_coluna1 = 'Semana Normalizado Z'
    df_rho = normaliza_dado(df_rho, df_padrao,coluna,   nova_coluna0,nova_coluna1)

    coluna = 'Int_fluxo_mes'
    nova_coluna0 = 'Mês Normalizado'
    nova_coluna1 = 'Mês Normalizado Z'
    df_rho = normaliza_dado(df_rho,  df_padrao, coluna, nova_coluna0,nova_coluna1)
    coluna = 'Int_fluxo_bimestre'
    nova_coluna0 = 'Bimestre Normalizado'
    nova_coluna1 = 'Bimestre Normalizado Z'
    df_rho = normaliza_dado(df_rho,  df_padrao,coluna,  nova_coluna0,nova_coluna1)
    coluna = 'Int_fluxo_trimestre'
    nova_coluna0 = 'Trimestre Normalizado'
    nova_coluna1 = 'Trimestre Normalizado Z'
    df_rho = normaliza_dado(df_rho, df_padrao, coluna,  nova_coluna0,nova_coluna1)
    coluna = 'Int_fluxo_semestre'
    nova_coluna0 = 'Semestre Normalizado'
    nova_coluna1 = 'Semestre Normalizado Z'
    df_rho = normaliza_dado(df_rho,  df_padrao, coluna, nova_coluna0,nova_coluna1)
    coluna = 'Int_fluxo_ano'
    nova_coluna0 = 'Ano Normalizado'
    nova_coluna1 = 'Ano Normalizado Z'
    df_rho = normaliza_dado(df_rho, df_padrao, coluna, nova_coluna0,nova_coluna1)
    return df_rho


def normaliza_dado2(df_rho,df_padrao,coluna0, coluna1, nova_coluna0,nova_coluna1):

    # (medido - padrão) / padrão Z-score
    df_rho[nova_coluna0] = df_rho[[coluna1]].values / df_padrao[[coluna0]].values
    df_rho[nova_coluna1] = (df_rho[[coluna1]].values - df_padrao[[coluna0]].values)/ df_padrao[[coluna0]].values
    return df_rho

def normaliza_int_flux_df2(df_rho,df_padrao):
    #todos de Int_flux
    #n_muons por n_int_flux
    coluna0 = 'n_Int_fluxo'
    
    coluna1 = 'n_Int_fluxo_hora'
    nova_coluna0 = 'Hora Normalizado'
    nova_coluna1 = 'Hora Normalizado Z'
    df_rho = normaliza_dado2(df_rho, df_padrao, coluna0, coluna1, nova_coluna0, nova_coluna1)

    coluna1 = 'n_Int_fluxo_dia'
    nova_coluna0 = 'Dia Normalizado'
    nova_coluna1 = 'Dia Normalizado Z'
    df_rho = normaliza_dado2(df_rho, df_padrao, coluna0, coluna1, nova_coluna0,nova_coluna1)

    coluna1 = 'n_Int_fluxo_semana'
    nova_coluna0 = 'Semana Normalizado'
    nova_coluna1 = 'Semana Normalizado Z'
    df_rho = normaliza_dado2(df_rho, df_padrao,coluna0, coluna1,   nova_coluna0,nova_coluna1)

    coluna1 = 'n_Int_fluxo_mes'
    nova_coluna0 = 'Mês Normalizado'
    nova_coluna1 = 'Mês Normalizado Z'
    df_rho = normaliza_dado2(df_rho, df_padrao, coluna0, coluna1, nova_coluna0, nova_coluna1)
    
    coluna1 = 'n_Int_fluxo_bimestre'
    nova_coluna0 = 'Bimestre Normalizado'
    nova_coluna1 = 'Bimestre Normalizado Z'
    df_rho = normaliza_dado2(df_rho, df_padrao, coluna0, coluna1, nova_coluna0, nova_coluna1)
    
    coluna1 = 'n_Int_fluxo_trimestre'
    nova_coluna0 = 'Trimestre Normalizado'
    nova_coluna1 = 'Trimestre Normalizado Z'
    df_rho = normaliza_dado2(df_rho, df_padrao, coluna0, coluna1,  nova_coluna0,nova_coluna1)
    
    coluna1 = 'n_Int_fluxo_semestre'
    nova_coluna0 = 'Semestre Normalizado'
    nova_coluna1 = 'Semestre Normalizado Z'
    df_rho = normaliza_dado2(df_rho,  df_padrao, coluna0, coluna1, nova_coluna0,nova_coluna1)
    
    coluna1 = 'n_Int_fluxo_ano'
    nova_coluna0 = 'Ano Normalizado'
    nova_coluna1 = 'Ano Normalizado Z'
    df_rho = normaliza_dado2(df_rho, df_padrao, coluna0, coluna1, nova_coluna0,nova_coluna1)
    return df_rho

#contagem dos muons
def calc_n_muons(df, coluna_int_flux, acceptance):
    df['n_' + coluna_int_flux] = df[coluna_int_flux]*acceptance
    #print("Espessura %.2f m, \n Intensidade de  %f s⁻¹GeV⁻¹"  %(Lm[3], df["Superficie_fluxo"][3] - df["Int_fluxo"][3]))

  #  print("Acceptance %.2f cm⁻²Sr⁻¹, \n Número de Muons %f cm⁻²Sr⁻¹s⁻¹GeV⁻¹"  %(acceptance, df["Superficie_fluxo"][3]*acceptance - df[coluna_int_flux][3]*acceptance))

              
    return df
              
def mesh_dado(coluna, df_are, df_shale, df_std, df_lime, df_gab):
    dados = np.array([
        df_are[coluna],
        df_shale[coluna],
        df_std[coluna],
        df_lime[coluna],
        df_gab[coluna]
    ])
    dados = dados.T
    plt.figure(figsize=(12, 6))

    plt.imshow(dados, cmap='Wistia', aspect='auto', interpolation='nearest')

    for y in range(dados.shape[0]):
        for x in range(dados.shape[1]):
            plt.text(x, y, f'{dados[y, x]:3.2}',
                     ha='center', va='center', color='black', fontsize=8)

    plt.xticks(np.arange(0, 5), ['Arenito', 'Folhelho', 'Rocha Padrão', 'Calcário', 'Gabro'])
#    plt.xticks(np.arange(0, 5),['2.357', '2.512', '2.65', '2.711', '3.156'])

    plt.yticks(np.arange(0, 14), df_std['Profundidade (m)'])

    plt.ylim(-0.5, dados.shape[0] - 0.5)
    plt.gca().invert_yaxis()

    plt.title(coluna, fontweight="bold")
    plt.ylabel('Comprimento (m)')
    plt.xlabel('Materiais')
    plt.colorbar(label='Dias')

    plt.show()
    return
              
def viabilidade_lesparre(df, var,df_std, cenith, acceptance):
    tempo_ = []
    
    for posicao in range(len(df)):
#        print(posicao)
        _L         = df['Profundidade (m)'][posicao]  # m                      
        _rho       = df['Densidade (g/cm³)'][posicao] # gcm-3
        _opacidade = df['Opacidade (g/cm²)'][posicao]     # m s.r.e
        _I_por_dia_variacao = df[var][posicao] # cm-2sr-1dia-1 
        _I_por_dia_referencia = df_std[var][posicao] # cm-2sr-1dia-1 
        _theta     = cenith   # graus
        _aceitacao = acceptance   # cm2sr
                                        
        c = 2    #o proprio raiz de N  c = 2* np.sqrt(Numero_muons), se np.sqrt(N) = 1, então c = 2.

        tempo = c * _I_por_dia_referencia / _aceitacao / (_I_por_dia_referencia - _I_por_dia_variacao)**2
        
        tempo_formatado = "{:.3f}".format(tempo)
        opacidade_formatado = "{:.0f}".format(_opacidade)

        _I_por_dia_variacao_formatado = "{:.3f}".format(_I_por_dia_variacao)
        _I_por_dia_referencia_formatado = "{:.3f}".format(_I_por_dia_referencia)
        #× tempo
        print(
            f"Rho: {_rho} g/cm³, L: {_L} m, Tempo: {tempo_formatado}, Tempo mínimo: {np.round(tempo)}, "
            f"Op: {opacidade_formatado}, I Delta dia: {_I_por_dia_variacao_formatado}, "
            f"I Referencia Dia: {_I_por_dia_referencia_formatado}"
        )
       # print("Rho:", _rho,"g/cm³","L", _L,"m", "Tempo", tempo_formatado, "Tempo mínimo:",np.round(tempo),"Op", opacidade_formatado, "I Delta dia", _I_por_dia_variacao_formatado, "I Referencia Dia",_I_por_dia_referencia_formatado)

        
        tempo_.append(tempo_formatado) #para um segundo NAO ESTÁ ARREDONDADO

    df["Tempo"] = tempo_

    return df