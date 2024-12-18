import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import streamlit as st
from sklearn.linear_model import LinearRegression

# Configuration de la page
st.set_page_config(page_title="Simulateur d'Investissement", layout="wide")

# Ajout de styles personnalisés
st.markdown("""
    <style>
        /* Style global de la barre latérale */
        [data-testid="stSidebar"] {
            background-color: #f5f5f5; /* Gris clair */
            border-radius: 10px;
            padding: 20px;
            border: 2px solid #D6A4A4; /* Contour rose pâle */
        }

        /* Libellés des champs */
        .stTextInput>label, .stNumberInput>label, .stDateInput>label, .stSelectbox>label {
            color: #0073E6; /* Bleu pour l'écriture */
            font-size: 0.9rem;
            font-weight: bold;
        }

        /* Champs de saisie */
        .stTextInput input, .stNumberInput input, .stDateInput input, .stSelectbox select {
            background-color: #ffffff; /* Blanc */
            border: 1px solid #D6A4A4; /* Contour rose pâle */
            border-radius: 8px;
            padding: 8px;
            font-size: 0.9rem;
        }

        /* Boutons */
        .stButton>button {
            background-color: #ffffff; /* Blanc */
            color: #0073E6; /* Texte bleu */
            border: 1px solid #D6A4A4; /* Contour rose pâle */
            border-radius: 8px;
            padding: 10px;
            font-size: 0.9rem;
            font-weight: bold;
            cursor: pointer;
        }

        .stButton>button:hover {
            background-color: #FDEDED; /* Rose clair */
            color: #004080; /* Bleu plus foncé */
        }

        /* Messages d'erreur ou d'alerte */
        .stAlert {
            background-color: #FFE6E6; /* Rose très clair */
            color: #CC0000; /* Rouge vif pour texte d'erreur */
            font-weight: bold;
            padding: 10px;
            border-radius: 8px;
        }

        /* Messages de bienvenue */
        .welcome-box {
            background-color: #ffffff; /* Blanc */
            border: 2px solid #D6A4A4; /* Contour rose pâle */
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .welcome-box h1 {
            color: #0073E6; /* Bleu */
            font-size: 2rem;
            margin-bottom: 10px;
        }
        .welcome-box p {
            color: #333333; /* Gris foncé pour le texte explicatif */
            font-size: 1rem;
            line-height: 1.5;
        }
    </style>
""", unsafe_allow_html=True)

# Message de bienvenue
st.markdown("""
    <div class="welcome-box">
        <h1>Bienvenue dans le Simulateur d'Investissement</h1>
    </div>
""", unsafe_allow_html=True)

# Paramètres de l'utilisateur
def sidebar_parameters():
    st.sidebar.header("Paramètres de l'investissement")

    actif = st.sidebar.text_input("Symbole de l'actif principal (ex: AAPL)", value="AAPL")
    autre_actif = st.sidebar.text_input("Symbole d'un autre actif (optionnel, ex: MSFT)", value="")
    date_debut = st.sidebar.date_input("Date de début", value=pd.to_datetime("2020-01-01"))
    date_fin = st.sidebar.date_input("Date de fin", value=pd.to_datetime("2024-10-25"))
    
    if date_debut >= date_fin:
        st.sidebar.error("La date de début doit précéder la date de fin.")

    taux_sans_risque = st.sidebar.number_input("Taux sans risque annuel (%)", min_value=0.0, value=2.0, step=0.1) / 100
    montant_initial = st.sidebar.number_input("Montant initial (€)", min_value=0, value=10000, step=1000)
    montant_contribution = st.sidebar.number_input(
        "Montant des contributions fréquentes (€)", min_value=0, value=500, step=100
    )
    frequence_contributions = st.sidebar.selectbox(
        "Fréquence des contributions",
        options=["Mensuelle", "Trimestrielle", "Semestrielle", "Annuelle"],
        index=0
    )
    frais_gestion = st.sidebar.number_input("Frais de gestion annuels (%)", min_value=0.0, value=0.50, step=0.05)

    return actif, autre_actif, date_debut, date_fin, taux_sans_risque, montant_initial, montant_contribution, frequence_contributions, frais_gestion

# Appel des paramètres
actif, autre_actif, date_debut, date_fin, taux_sans_risque, montant_initial, montant_contribution, frequence_contributions, frais_gestion = sidebar_parameters()


# Fonction pour vérifier la validité du symbole
def verifier_symbole(actif):
    try:
        donnees_brutes = yf.download(actif, start=str(date_debut), end=str(date_fin))
        if donnees_brutes.empty:
            raise ValueError("Données vides, symbole peut-être invalide.")
        return donnees_brutes
    except Exception as e:
        st.error(f"Erreur : Impossible de trouver l'actif '{actif}'. Veuillez vérifier le symbole.")
        return None

# Vérifier si les symboles sont valides
donnees_brutes = verifier_symbole(actif)
donnees_autre_actif = None

# Si un second actif est saisi, vérifier également
if autre_actif:
    donnees_autre_actif = verifier_symbole(autre_actif)

# Vérifier que le premier actif est valide, sinon afficher uniquement le message d'erreur
if donnees_brutes is None:
    st.stop()  # Arrête l'exécution du script après avoir affiché l'erreur pour le premier actif

# Traitement pour le premier actif
donnees = donnees_brutes[['Adj Close']].copy()
donnees.columns = ['Prix Ajusté']
donnees['Rendement Quotidien'] = donnees['Prix Ajusté'].pct_change()
donnees['Rendement Cumulé'] = (1 + donnees['Rendement Quotidien']).cumprod()

volatilite_portefeuille = donnees['Rendement Quotidien'].std() * np.sqrt(252)
rendement_portefeuille = donnees['Rendement Quotidien'].mean() * 252
ratio_sharpe = (rendement_portefeuille - taux_sans_risque) / volatilite_portefeuille
valeur_initiale = donnees['Prix Ajusté'].iloc[0]
valeur_finale = donnees['Prix Ajusté'].iloc[-1]
rendement_total = ((valeur_finale - valeur_initiale) / valeur_initiale) * 100
nombre_annees = (donnees.index[-1] - donnees.index[0]).days / 365.25
cagr = ((valeur_finale / valeur_initiale) ** (1 / nombre_annees) - 1) * 100

# Affichage des métriques pour un seul actif ou deux actifs
st.write("### Analyse des performances")


# Vérification si le second actif est présent
if donnees_autre_actif is not None and not donnees_autre_actif.empty:
    donnees_autre = donnees_autre_actif[['Adj Close']].copy()
    donnees_autre.columns = ['Prix Ajusté']
    donnees_autre['Rendement Quotidien'] = donnees_autre['Prix Ajusté'].pct_change()
    donnees_autre['Rendement Cumulé'] = (1 + donnees_autre['Rendement Quotidien']).cumprod()

    # Calcul des métriques pour le second actif
    volatilite_autre = donnees_autre['Rendement Quotidien'].std() * np.sqrt(252)
    rendement_autre = donnees_autre['Rendement Quotidien'].mean() * 252
    ratio_sharpe_autre = (rendement_autre - taux_sans_risque) / volatilite_autre
    valeur_initiale_autre = donnees_autre['Prix Ajusté'].iloc[0]
    valeur_finale_autre = donnees_autre['Prix Ajusté'].iloc[-1]
    rendement_total_autre = ((valeur_finale_autre - valeur_initiale_autre) / valeur_initiale_autre) * 100
    cagr_autre = ((valeur_finale_autre / valeur_initiale_autre) ** (1 / nombre_annees) - 1) * 100
else:
    donnees_autre = None  # Assurez-vous que cette variable est définie



if donnees_autre is not None:
    st.markdown(f"""
    <table style="width:100%; border-collapse: collapse; text-align: center;">
        <tr>
            <th style="background-color:#0073E6; color:white;">Métriques</th>
            <th style="background-color:#0073E6; color:white;">{actif.upper()}</th>
            <th style="background-color:#0073E6; color:white;">{autre_actif.upper()}</th>
        </tr>
        <tr><td>Volatilité annualisée</td>
            <td style="color:#0073E6; font-weight:bold;">{volatilite_portefeuille:.2%}</td>
            <td style="color:#0073E6; font-weight:bold;">{volatilite_autre:.2%}</td></tr>
        <tr><td>Ratio de Sharpe</td>
            <td style="color:#0073E6; font-weight:bold;">{ratio_sharpe:.2f}</td>
            <td style="color:#0073E6; font-weight:bold;">{ratio_sharpe_autre:.2f}</td></tr>
        <tr><td>Rendement total</td>
            <td style="color:#0073E6; font-weight:bold;">{rendement_total:.2f}%</td>
            <td style="color:#0073E6; font-weight:bold;">{rendement_total_autre:.2f}%</td></tr>
        <tr><td>CAGR</td>
            <td style="color:#0073E6; font-weight:bold;">{cagr:.2f}%</td>
            <td style="color:#0073E6; font-weight:bold;">{cagr_autre:.2f}%</td></tr>
    </table>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <table style="width:50%; border-collapse: collapse; text-align: center; margin: auto;">
        <tr>
            <th style="background-color:#0073E6; color:white;">Métriques</th>
            <th style="background-color:#0073E6; color:white;">{actif.upper()}</th>
        </tr>
        <tr><td>Volatilité annualisée</td>
            <td style="color:#0073E6; font-weight:bold;">{volatilite_portefeuille:.2%}</td></tr>
        <tr><td>Ratio de Sharpe</td>
            <td style="color:#0073E6; font-weight:bold;">{ratio_sharpe:.2f}</td></tr>
        <tr><td>Rendement total</td>
            <td style="color:#0073E6; font-weight:bold;">{rendement_total:.2f}%</td></tr>
        <tr><td>CAGR</td>
            <td style="color:#0073E6; font-weight:bold;">{cagr:.2f}%</td></tr>
    </table>
    """, unsafe_allow_html=True)



# Graphique comparatif des rendements cumulés
st.write("### Comparaison des rendements cumulés")
fig, ax = plt.subplots(figsize=(12, 6))

# Tracer les rendements cumulés pour chaque actif
ax.plot(donnees.index, donnees['Rendement Cumulé'], label=f"Premier Actif : {actif.upper()}", color='blue')
if donnees_autre_actif is not None:
    ax.plot(donnees_autre.index, donnees_autre['Rendement Cumulé'], label=f"Second Actif : {autre_actif.upper()}", color='green')

# Ajouter des étiquettes et une légende
ax.set_xlabel('Date')
ax.set_ylabel('Rendement Cumulé')
ax.legend()

# Afficher le graphique
st.pyplot(fig)



 # Calcul des rendements mensuels
donnees['Mois'] = donnees.index.to_period('M')  # Regrouper par mois
donnees_mensuelles = donnees.groupby('Mois').last()  # Prendre les dernières valeurs par mois
donnees_mensuelles['Rendement Mensuel'] = donnees_mensuelles['Prix Ajusté'].pct_change()

# Recalculer les prix en fonction de la fréquence choisie
frequences = {
    'Mensuelle': 'M',
    'Trimestrielle': '3M',
    'Semestrielle': '6M',
    'Annuelle': '12M'
}

if frequence_contributions in frequences:
    prix_par_periode = donnees.resample(frequences[frequence_contributions]).last()['Prix Ajusté']
else:
    st.error("Fréquence invalide !")
    prix_par_periode = pd.Series(dtype=float)

# Calcul des rendements pour la fréquence choisie
rendements_frequents = prix_par_periode.pct_change().dropna()


# Comparaison des deux actifs
if donnees_autre_actif is not None and not donnees_autre_actif.empty:
   
    # Traitement des données pour le second actif
    donnees_autre = donnees_autre_actif[['Adj Close']].copy()
    donnees_autre.columns = ['Prix Ajusté']
    donnees_autre['Rendement Quotidien'] = donnees_autre['Prix Ajusté'].pct_change()
    donnees_autre['Rendement Cumulé'] = (1 + donnees_autre['Rendement Quotidien']).cumprod()

 # Histogramme(s) des rendements avec barres positives en vert et négatives en rouge
if donnees_autre_actif is not None and not donnees_autre_actif.empty:

    st.write(f"### Distribution des rendements ({frequence_contributions} - {actif.upper()} et {autre_actif.upper()})")
    col1, col2 = st.columns(2)

    # Premier actif
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        rendements = donnees['Rendement Quotidien'].dropna()

        # Séparation des rendements positifs et négatifs
        positif = rendements[rendements >= 0]
        negatif = rendements[rendements < 0]

        # Tracer les rendements positifs (vert) et négatifs (rouge)
        ax.hist(positif, bins=20, alpha=0.7, label='Positifs', color='green', edgecolor='black')
        ax.hist(negatif, bins=20, alpha=0.7, label='Négatifs', color='red', edgecolor='black')

        # Ajouter le symbole en haut à gauche avec encadrement
        ax.text(0.02, 0.98, f"{actif.upper()}",
                transform=ax.transAxes, fontsize=10, fontweight='normal',
                color='black', ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        ax.set_xlabel("Rendement")
        ax.set_ylabel("Fréquence")
        ax.legend()
        st.pyplot(fig)

    # Deuxième actif
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        rendements_autre = donnees_autre['Rendement Quotidien'].dropna()

        # Séparation des rendements positifs et négatifs
        positif_autre = rendements_autre[rendements_autre >= 0]
        negatif_autre = rendements_autre[rendements_autre < 0]

        # Tracer les rendements positifs (vert) et négatifs (rouge)
        ax.hist(positif_autre, bins=20, alpha=0.7, label='Positifs', color='green', edgecolor='black')
        ax.hist(negatif_autre, bins=20, alpha=0.7, label='Négatifs', color='red', edgecolor='black')

        # Ajouter le symbole en haut à gauche avec encadrement
        ax.text(0.02, 0.98, f"{autre_actif.upper()}",
                transform=ax.transAxes, fontsize=10, fontweight='normal',
                color='black', ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        ax.set_xlabel("Rendement")
        ax.set_ylabel("Fréquence")
        ax.legend()
        st.pyplot(fig)

else:
    # Histogramme pour un seul actif
    st.write(f"### Distribution des rendements ({frequence_contributions} - {actif.upper()})")
    fig, ax = plt.subplots(figsize=(10, 5))
    rendements = donnees['Rendement Quotidien'].dropna()

    # Séparation des rendements positifs et négatifs
    positif = rendements[rendements >= 0]
    negatif = rendements[rendements < 0]

    # Tracer les rendements positifs (vert) et négatifs (rouge)
    ax.hist(positif, bins=20, alpha=0.7, label='Positifs', color='green', edgecolor='black')
    ax.hist(negatif, bins=20, alpha=0.7, label='Négatifs', color='red', edgecolor='black')

    # Ajouter le symbole en haut à gauche avec encadrement
    ax.text(0.02, 0.98, f"{actif.upper()}",
            transform=ax.transAxes, fontsize=10, fontweight='normal',
            color='black', ha='left', va='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    ax.set_xlabel("Rendement")
    ax.set_ylabel("Fréquence")
    ax.legend()
    st.pyplot(fig)


# Visualisation de la volatilité avec boîte à moustaches
st.write(f"### Volatilité des rendements ({frequence_contributions})")

if donnees_autre_actif is not None and not donnees_autre_actif.empty:
    # Deux boîtes à moustaches côte à côte
    col1, col2 = st.columns(2)

    # Boîte à moustaches pour le premier actif
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(rendements_frequents, vert=False, patch_artist=True,
                   boxprops=dict(facecolor='blue', color='black'),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='black'),
                   medianprops=dict(color='red'))

        # Ajouter le texte encadré à l'intérieur en haut à gauche
        ax.text(0.02, 0.98, f"({actif.upper()})", transform=ax.transAxes,
                fontsize=10, ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        ax.set_xlabel("Rendement")
        st.pyplot(fig)

    # Boîte à moustaches pour le deuxième actif
    with col2:
        # Calculer les rendements pour le deuxième actif
        prix_par_periode_autre = donnees_autre.resample(frequences[frequence_contributions]).last()['Prix Ajusté']
        rendements_frequents_autre = prix_par_periode_autre.pct_change().dropna()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(rendements_frequents_autre, vert=False, patch_artist=True,
                   boxprops=dict(facecolor='green', color='black'),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='black'),
                   medianprops=dict(color='red'))

        # Ajouter le texte encadré à l'intérieur en haut à gauche
        ax.text(0.02, 0.98, f"({autre_actif.upper()})", transform=ax.transAxes,
                fontsize=10, ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        ax.set_xlabel("Rendement")
        st.pyplot(fig)
else:
    # Une seule boîte à moustaches si aucun deuxième actif
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(rendements_frequents, vert=False, patch_artist=True,
               boxprops=dict(facecolor='blue', color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               medianprops=dict(color='red'))

    # Ajouter le texte encadré à l'intérieur en haut à gauche
    ax.text(0.02, 0.98, f"({actif.upper()})", transform=ax.transAxes,
            fontsize=10, ha='left', va='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    ax.set_xlabel("Rendement")
    st.pyplot(fig)


# Téléchargement des données de l'indice ACWI IMI via yfinance
try:
    symbole_acwi = "ACWI"  # Symbole pour l'indice ACWI IMI
    donnees_acwi_brutes = yf.download(symbole_acwi, start=str(date_debut), end=str(date_fin))
    if not donnees_acwi_brutes.empty:
        donnees_acwi = donnees_acwi_brutes[['Adj Close']].copy()
        donnees_acwi.columns = ['Prix Ajusté']
        donnees_acwi['Rendement Quotidien'] = donnees_acwi['Prix Ajusté'].pct_change()
        donnees_acwi['Rendement Cumulé'] = (1 + donnees_acwi['Rendement Quotidien']).cumprod()
    else:
        st.warning("Les données pour l'indice ACWI IMI sont vides.")
        donnees_acwi = None
except Exception as e:
    st.error(f"Erreur lors du téléchargement des données de l'indice ACWI IMI : {e}")
    donnees_acwi = None

# Comparaison avec l'indice ACWI IMI
if donnees_acwi is not None:
    st.write("### Comparaison avec l'indice ACWI IMI")
    
    # Si un deuxième actif est saisi, afficher deux graphiques côte à côte
    if donnees_autre_actif is not None and not donnees_autre_actif.empty:
        col1, col2 = st.columns(2)
        
        # Premier graphique : Comparaison du premier actif avec ACWI IMI
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(donnees.index, donnees['Rendement Cumulé'], label=f'Portefeuille ({actif.upper()})', color='blue')
            ax.plot(donnees_acwi.index, donnees_acwi['Rendement Cumulé'], label='Indice ACWI IMI', color='orange')
            ax.set_xlabel('Date')
            ax.set_ylabel('Rendement Cumulatif')
            ax.legend()
            st.pyplot(fig)
        
        # Deuxième graphique : Comparaison du second actif avec ACWI IMI
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(donnees_autre.index, donnees_autre['Rendement Cumulé'], label=f'Portefeuille ({autre_actif.upper()})', color='green')
            ax.plot(donnees_acwi.index, donnees_acwi['Rendement Cumulé'], label='Indice ACWI IMI', color='orange')
            ax.set_xlabel('Date')
            ax.set_ylabel('Rendement Cumulatif')
            ax.legend()
            st.pyplot(fig)
    else:
        # Un seul graphique si aucun autre actif n'est saisi
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(donnees.index, donnees['Rendement Cumulé'], label=f'Portefeuille ({actif.upper()})', color='blue')
        ax.plot(donnees_acwi.index, donnees_acwi['Rendement Cumulé'], label='Indice ACWI IMI', color='orange')
        ax.set_xlabel('Date')
        ax.set_ylabel('Rendement Cumulatif')
        ax.legend()
        st.pyplot(fig)
else:
    st.warning("Les données de l'indice ACWI IMI ne sont pas disponibles pour effectuer la comparaison.")


# Simulation Lump Sum
donnees['Valeur Lump Sum'] = montant_initial * donnees['Rendement Cumulé']

# Simulation DCA
contributions_dca = {
    'Mensuelle': montant_contribution,
    'Trimestrielle': montant_contribution * 3,
    'Semestrielle': montant_contribution * 6,
    'Annuelle': montant_contribution * 12
}

def calcul_dca(prix_par_periode, contribution):
    valeur_actuelle = 0
    valeurs = []
    for prix in prix_par_periode:
        valeur_actuelle += contribution / prix
        valeurs.append(valeur_actuelle * prix)
    return valeurs

portefeuille_dca = calcul_dca(prix_par_periode, contributions_dca[frequence_contributions])
dca_df = pd.DataFrame({'Prix': prix_par_periode, 'Valeur Portefeuille DCA': portefeuille_dca}, index=prix_par_periode.index)

# Comparaison des stratégies Lump Sum et DCA pour le premier actif
st.write("### Comparaison des stratégies d'investissement : Lump Sum vs DCA")

if donnees_autre_actif is not None and not donnees_autre_actif.empty:
    # Calcul des prix par période pour le deuxième actif
    prix_par_periode_autre = donnees_autre.resample(frequences[frequence_contributions]).last()['Prix Ajusté']
    portefeuille_dca_autre = calcul_dca(prix_par_periode_autre, contributions_dca[frequence_contributions])
    dca_df_autre = pd.DataFrame({'Prix': prix_par_periode_autre, 
                                 'Valeur Portefeuille DCA': portefeuille_dca_autre}, 
                                 index=prix_par_periode_autre.index)

    # Création de deux colonnes pour les graphiques
    col1, col2 = st.columns(2)

    # Premier actif : Lump Sum vs DCA
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(donnees.index, donnees['Valeur Lump Sum'], label='Lump Sum', color='blue')
        ax.plot(dca_df.index, dca_df['Valeur Portefeuille DCA'], label=f'DCA {frequence_contributions}', color='green')

        # Ajouter le symbole en haut à gauche
        ax.text(0.02, 0.98, f"{actif.upper()}",
                transform=ax.transAxes, fontsize=10, fontweight='normal',
                color='black', ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        ax.set_xlabel("Date")
        ax.set_ylabel("Valeur du Portefeuille (€)")
        ax.legend()
        st.pyplot(fig)

    # Deuxième actif : Lump Sum vs DCA
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(donnees_autre.index, montant_initial * donnees_autre['Rendement Cumulé'], label='Lump Sum', color='blue')
        ax.plot(dca_df_autre.index, dca_df_autre['Valeur Portefeuille DCA'], label=f'DCA {frequence_contributions}', color='green')

        # Ajouter le symbole en haut à gauche
        ax.text(0.02, 0.98, f"{autre_actif.upper()}",
                transform=ax.transAxes, fontsize=10, fontweight='normal',
                color='black', ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        ax.set_xlabel("Date")
        ax.set_ylabel("Valeur du Portefeuille (€)")
        ax.legend()
        st.pyplot(fig)

else:
    # Si aucun deuxième actif n'est saisi, afficher uniquement le premier graphique
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(donnees.index, donnees['Valeur Lump Sum'], label='Lump Sum', color='blue')
    ax.plot(dca_df.index, dca_df['Valeur Portefeuille DCA'], label=f'DCA {frequence_contributions}', color='green')

    # Ajouter le symbole en haut à gauche
    ax.text(0.02, 0.98, f"{actif.upper()}",
            transform=ax.transAxes, fontsize=10, fontweight='normal',
            color='black', ha='left', va='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur du Portefeuille (€)")
    ax.legend()
    st.pyplot(fig)

# Affichage des résultats finaux
st.write("## Résultats Finaux")
st.write(f"**Valeur finale Lump Sum :** {donnees['Valeur Lump Sum'].iloc[-1]:.2f} €")
st.write(f"**Valeur finale DCA ({frequence_contributions}) :** {dca_df['Valeur Portefeuille DCA'].iloc[-1]:.2f} €")

# Calcul des résultats pour le premier actif
montant_final_lump_sum = donnees['Valeur Lump Sum'].iloc[-1]
montant_final_dca = dca_df['Valeur Portefeuille DCA'].iloc[-1]
gain_realise_lump_sum = montant_final_lump_sum - montant_initial
gain_realise_dca = montant_final_dca - montant_initial

# Calcul du rendement et des autres métriques pour le premier actif
duree_investissement = (donnees.index[-1] - donnees.index[0]).days / 365

# Calcul du rendement annuel moyen (CAGR)
cagr_lump_sum = ((montant_final_lump_sum / montant_initial) ** (1 / duree_investissement) - 1) * 100
cagr_dca = ((montant_final_dca / montant_initial) ** (1 / duree_investissement) - 1) * 100

# Moyenne des Contributions pour DCA
moyenne_contributions_dca = montant_contribution * (12 if frequence_contributions == "Mensuelle" else 1)

# Tableau comparatif pour le premier actif
tableau_resultats = pd.DataFrame({
    "Stratégie": ["Lump Sum", f"DCA ({frequence_contributions})"],
    "Montant Initial (€)": [montant_initial, montant_initial],
    "Montant Final (€)": [montant_final_lump_sum, montant_final_dca],
    "Gains Réalisés (€)": [gain_realise_lump_sum, gain_realise_dca],
    "Rendement Annuel Moyen (%)": [cagr_lump_sum, cagr_dca],
    "Moyenne Contributions (€)": ["N/A", moyenne_contributions_dca]
})

# Style personnalisé pour l'en-tête bleu
def header_style():
    return [
        {'selector': 'thead th', 'props': [('background-color', '#0073E6'),
                                           ('color', 'white'),
                                           ('font-weight', 'bold'),
                                           ('text-align', 'center')]},
    ]

# Fonction pour appliquer la couleur aux gains/pertes
def highlight_table(valeur):
    if isinstance(valeur, str):  # Garder les colonnes textuelles neutres
        return ''
    elif valeur < 0:  # Couleur rouge pour les pertes
        return 'color: red; font-weight: bold;'
    else:  # Couleur verte pour les gains
        return 'color: green; font-weight: bold;'

# Transposer le tableau pour l'affichage
tableau_resultats_transpose = tableau_resultats.T
tableau_resultats_transpose.columns = tableau_resultats_transpose.iloc[0]  # Prendre la première ligne comme en-têtes
tableau_resultats_transpose = tableau_resultats_transpose[1:]  # Supprimer l'ancienne ligne des en-têtes

# Application des styles avec l'en-tête en bleu
styled_table = (
    tableau_resultats_transpose.style
    .applymap(highlight_table)  # Mise en forme des gains/pertes
    .set_table_styles(header_style())  # Style de l'en-tête en bleu
    .format({
        "Montant Initial (€)": "{:.2f}",
        "Montant Final (€)": "{:.2f}",
        "Gains Réalisés (€)": "{:.2f}",
        "Rendement Annuel Moyen (%)": "{:.2f} %",
        "Moyenne Contributions (€)": "{:.2f}"
    })
)

# Affichage dans Streamlit
st.write("### Tableau comparatif des résultats d'investissement")
st.table(styled_table)


# Régression linéaire pour le premier actif
donnees['Jours'] = (donnees.index - donnees.index[0]).days
X = donnees['Jours'].values.reshape(-1, 1)
y = donnees['Prix Ajusté'].values.reshape(-1, 1)

modele = LinearRegression()
modele.fit(X, y)

# Prédictions et bandes d'incertitude pour le premier actif
donnees['Prix Prévu'] = modele.predict(X)
donnees['Résidus'] = donnees['Prix Ajusté'] - donnees['Prix Prévu']
ecart_type_residus = donnees['Résidus'].std()

for i in range(1, 4):  # ±1, ±2, ±3 écarts types
    donnees[f'Limite Supérieure (±{i})'] = donnees['Prix Prévu'] + i * ecart_type_residus
    donnees[f'Limite Inférieure (±{i})'] = donnees['Prix Prévu'] - i * ecart_type_residus

jours_futurs = np.arange(donnees['Jours'].max() + 1, donnees['Jours'].max() + 181).reshape(-1, 1)
previsions_futures = modele.predict(jours_futurs)
dates_futures = pd.date_range(start=donnees.index[-1] + pd.Timedelta(days=1), periods=180, freq='D')
donnees_futures = pd.DataFrame({
    'Prix Prévu': previsions_futures.flatten(),
    **{f'Limite Supérieure (±{i})': previsions_futures.flatten() + i * ecart_type_residus for i in range(1, 4)},
    **{f'Limite Inférieure (±{i})': previsions_futures.flatten() - i * ecart_type_residus for i in range(1, 4)}
}, index=dates_futures)

donnees_total = pd.concat([donnees, donnees_futures])

# Régression linéaire pour le deuxième actif (si saisi)
if donnees_autre_actif is not None and not donnees_autre_actif.empty:
    donnees_autre['Jours'] = (donnees_autre.index - donnees_autre.index[0]).days
    X_autre = donnees_autre['Jours'].values.reshape(-1, 1)
    y_autre = donnees_autre['Prix Ajusté'].values.reshape(-1, 1)

    modele_autre = LinearRegression()
    modele_autre.fit(X_autre, y_autre)

    donnees_autre['Prix Prévu'] = modele_autre.predict(X_autre)
    donnees_autre['Résidus'] = donnees_autre['Prix Ajusté'] - donnees_autre['Prix Prévu']
    ecart_type_residus_autre = donnees_autre['Résidus'].std()

    for i in range(1, 4):  # ±1, ±2, ±3 écarts types
        donnees_autre[f'Limite Supérieure (±{i})'] = donnees_autre['Prix Prévu'] + i * ecart_type_residus_autre
        donnees_autre[f'Limite Inférieure (±{i})'] = donnees_autre['Prix Prévu'] - i * ecart_type_residus_autre

    jours_futurs_autre = np.arange(donnees_autre['Jours'].max() + 1, donnees_autre['Jours'].max() + 181).reshape(-1, 1)
    previsions_futures_autre = modele_autre.predict(jours_futurs_autre)
    dates_futures_autre = pd.date_range(start=donnees_autre.index[-1] + pd.Timedelta(days=1), periods=180, freq='D')
    donnees_futures_autre = pd.DataFrame({
        'Prix Prévu': previsions_futures_autre.flatten(),
        **{f'Limite Supérieure (±{i})': previsions_futures_autre.flatten() + i * ecart_type_residus_autre for i in range(1, 4)},
        **{f'Limite Inférieure (±{i})': previsions_futures_autre.flatten() - i * ecart_type_residus_autre for i in range(1, 4)}
    }, index=dates_futures_autre)

    donnees_total_autre = pd.concat([donnees_autre, donnees_futures_autre])

# Affichage des graphiques de régression linéaire
if donnees_autre_actif is not None and not donnees_autre_actif.empty:
    st.write("### Régression linéaire pour prédire les rendements futurs")
    col1, col2 = st.columns(2)

    # Premier actif
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(donnees_total.index, donnees_total['Prix Ajusté'], label='Prix réel', color='blue')
        ax.plot(donnees_total.index, donnees_total['Prix Prévu'], label='Prix prédit', color='green')
        for i, color in zip(range(1, 4), ['green', 'orange', 'red']):
            ax.fill_between(donnees_total.index, donnees_total[f'Limite Inférieure (±{i})'], donnees_total[f'Limite Supérieure (±{i})'], color=color, alpha=0.2)
        ax.text(0.02, 0.98, f"{actif.upper()}",
                transform=ax.transAxes, fontsize=10, fontweight='normal',
                color='black', ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix Ajusté')
        ax.legend()
        st.pyplot(fig)

    # Deuxième actif
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(donnees_total_autre.index, donnees_total_autre['Prix Ajusté'], label='Prix réel', color='blue')
        ax.plot(donnees_total_autre.index, donnees_total_autre['Prix Prévu'], label='Prix prédit', color='green')
        for i, color in zip(range(1, 4), ['green', 'orange', 'red']):
            ax.fill_between(donnees_total_autre.index, donnees_total_autre[f'Limite Inférieure (±{i})'], donnees_total_autre[f'Limite Supérieure (±{i})'], color=color, alpha=0.2)
        ax.text(0.02, 0.98, f"{autre_actif.upper()}",
                transform=ax.transAxes, fontsize=10, fontweight='normal',
                color='black', ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix Ajusté')
        ax.legend()
        st.pyplot(fig)
else:
    st.write(f"### Régression linéaire ({actif.upper()}) pour prédire les rendements futurs")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(donnees_total.index, donnees_total['Prix Ajusté'], label='Prix réel', color='blue')
    ax.plot(donnees_total.index, donnees_total['Prix Prévu'], label='Prix prédit', color='green')
    for i, color in zip(range(1, 4), ['green', 'orange', 'red']):
        ax.fill_between(donnees_total.index, donnees_total[f'Limite Inférieure (±{i})'], donnees_total[f'Limite Supérieure (±{i})'], color=color, alpha=0.2)
    ax.text(0.02, 0.98, f"{actif.upper()}",
            transform=ax.transAxes, fontsize=10, fontweight='normal',
            color='black', ha='left', va='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.set_xlabel('Date')
    ax.set_ylabel('Prix Ajusté')
    ax.legend()
    st.pyplot(fig)
# Fonction pour créer un PDF
def creer_pdf():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Chemin vers la police Arial
    font_path = r'C:\Windows\Fonts\arial.ttf'  # Remplacez par le chemin correct

    # Utiliser une police Unicode
    pdf.add_font('Arial', '', font_path, uni=True)
    pdf.set_font("Arial", size=12)

    # Titre
    pdf.cell(200, 10, txt="Rapport d'Analyse d'Investissement", ln=True, align='C')

    # Ajout des métriques
    pdf.set_font("Arial", size=10)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Volatilité annualisée: {volatilite_portefeuille:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"Ratio de Sharpe: {ratio_sharpe:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Rendement total: {rendement_total:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"CAGR: {cagr:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Valeur finale Lump Sum: {donnees['Valeur Lump Sum'].iloc[-1]:.2f} €", ln=True)
    pdf.cell(200, 10, txt=f"Valeur finale DCA {frequence_contributions}: {dca_df['Valeur Portefeuille DCA'].iloc[-1]:.2f} €", ln=True)

    # Graphiques sous forme d'images
    for fig_num, fig in enumerate(plt.get_fignums(), start=1):
        plt.figure(fig)
        plt.savefig(f"figure_{fig_num}.png")
        pdf.add_page()
        pdf.image(f"figure_{fig_num}.png", x=10, y=30, w=180)

    # Sauvegarde temporaire
    pdf_path = "rapport_analyse.pdf"
    pdf.output(pdf_path)
    return pdf_path


# Bouton Streamlit pour exporter en PDF
if st.button("Exporter en PDF"):
    pdf_path = creer_pdf()
    with open(pdf_path, "rb") as file:
        st.download_button(
            label="Télécharger le rapport PDF",
            data=file,
            file_name="rapport_analyse.pdf",
            mime="application/pdf"
        )