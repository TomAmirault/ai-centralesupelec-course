import pandas as pd

# Charger le fichier CSV
df = pd.read_csv('ParticipantsCorbevilleTOSS2025_no_duplicates.csv', header=None, encoding='latin-1')

# Afficher les premières lignes pour vérifier
print(df.head())

# Diviser la première colonne en plusieurs nouvelles colonnes
splitted_columns = df[0].str.split(',', expand=True)

# Vérifier combien de colonnes ont été générées
print(f"Nombre de colonnes générées : {splitted_columns.shape[1]}")

# Si le nombre de colonnes générées est bien 4, nommer les nouvelles colonnes
if splitted_columns.shape[1] == 4:
    splitted_columns.columns = ['firstname', 'lastname', 'name', 'sport']
else:
    print("Le nombre de colonnes générées n'est pas égal à 4.")

# Fusionner les nouvelles colonnes avec le reste du DataFrame (s'il y a d'autres colonnes)
df = pd.concat([splitted_columns, df.iloc[:, 1:]], axis=1)

# Sauvegarder le fichier modifié
df.to_csv('ParticipantsCorbevilleTOSS2025_separated.csv', index=False)

print("Les colonnes ont été séparées et sauvegardées sous 'ParticipantsCorbevilleTOSS2025_separated.csv'.")


"""
import pandas as pd

# Charger le fichier CSV
df = pd.read_csv('ParticipantsCorbevilleTOSS2025.csv', encoding='latin-1')

# Supprimer les doublons dans le DataFrame
df = df.drop_duplicates()

# Sauvegarder le fichier sans doublons
df.to_csv('ParticipantsCorbevilleTOSS2025_no_duplicates.csv', index=False)

print("Les doublons ont été supprimés et le fichier a été sauvegardé sous 'ParticipantsCorbevilleTOSS2025_no_duplicates.csv'.")
"""