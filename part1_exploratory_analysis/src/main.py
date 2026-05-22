import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

file_path = "players_22.csv"
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 500
sns.set_palette("tab10")

df_players = pd.read_csv(file_path)

df_players_numeric = df_players.select_dtypes(include=['number'])
df_stats = pd.DataFrame([
    df_players_numeric.mean(),
    df_players_numeric.median(),
    df_players_numeric.min(),
    df_players_numeric.max(),
    df_players_numeric.std(),
    df_players_numeric.quantile(0.05),
    df_players_numeric.quantile(0.95),
    df_players_numeric.isna().sum()],
      index=[
    "średnia", 
    "mediana", 
    "min", "max", 
    "odchylenie standardowe", 
    "5-ty percentyl", 
    "95-ty percentyl", 
    "brakujące wartości"])
df_stats = df_stats.T
df_stats.to_csv("numeryczne.csv")

df_players_nonumeric = df_players.select_dtypes(exclude=['number'])
df_nonumeric_stats = pd.DataFrame([
    df_players_nonumeric.nunique(),
    df_players_nonumeric.isna().sum(),
    df_players_nonumeric.apply(lambda x: x.value_counts(normalize=True).to_dict())],
      index=[
    "liczba unikalnych klas", 
    "brakujące wartości", 
    "proporcja klas"])
df_nonumeric_stats = df_nonumeric_stats.T
df_nonumeric_stats.to_csv("nienumeryczne.csv")


#Boxplot
sns.catplot(
    data=df_players, 
    x="nationality_name", 
    y="overall", 
    kind="box", 
    order=["Poland", "Spain"],
    linewidth = 2,
    fliersize = 5)
plt.xlabel("Narodowość")
plt.ylabel("Ocena ogólna")
plt.title("Wykres oceny ogólnej piłkarza w zależności od narodowości")
plt.savefig("boxplot.png", bbox_inches="tight")
plt.close()


#Violinbox
sns.catplot(data=df_players, x="league_level", y="skill_ball_control", kind="violin", bw_adjust=0.5)
plt.xlabel("Stopień rogrywkowy")
plt.ylabel("Kontrola piłki")
plt.title("Wykres umiejętności kontroli piłki względem stopnia rozgrywkowego")
plt.savefig("violinplot.png", bbox_inches="tight")
plt.close()


#Error bars
df_pom = df_players[df_players["club_position"].isin(["CB", "LW"])]

sns.stripplot(data=df_pom, y="passing", hue="club_position", palette="pastel")
plt.ylabel("Umiejętność podawania")
plt.legend().set_title("Pozycja")
plt.ylim(30, 90)
plt.savefig("errorbars1.png", bbox_inches="tight")
plt.close()

sns.pointplot(data=df_pom, x="club_position", y="passing" , hue="club_position", errorbar=("sd"), join=False,
                palette="pastel", capsize=0.2)
plt.ylabel("Umiejętność podawania")
plt.xlabel("Pozycja")
plt.ylim(30, 90)
plt.savefig("errorbars2.png", bbox_inches="tight")
plt.close()


#Histogram
sns.histplot(df_players["club_jersey_number"], bins=100, kde=True, discrete=True)
plt.title("Histogram dla liczby osób z danym numerem kosuzlki")
plt.xlabel("Numer koszulki klubowej")
plt.ylabel("Liczba zawodników")
plt.savefig("histogram.png", bbox_inches="tight")
plt.close()


#Histogram warunkowy
wykres = sns.histplot(df_players, x="age", hue="preferred_foot", bins=100, kde=True, discrete=True, palette="viridis")
plt.title("Histogram dla wieku")
plt.xlabel("Wiek")
plt.ylabel("Liczba zawodników")
legend = plt.gca().get_legend()
legend.set_title('Lepsza noga')
plt.savefig("histogramwarunkowy.png", bbox_inches="tight")
plt.close()


#Heatmapa
corr_matrix = df_players[["pace", "height_cm", "weight_kg", "age", "movement_agility", "value_eur", "wage_eur"]].corr()
hm = sns.heatmap(data=corr_matrix,
            vmin=-1,
            vmax=1, 
            cmap="afmhot_r", 
            center=0, 
            annot=True, 
            fmt='.2f',  
            linewidths=0, 
            linecolor='black')
hm.set_xticklabels(["Szybkość", "Wzrost", "Waga", "Wiek", "Zwinność", "Wartość", "Pensja"])
hm.set_yticklabels(["Szybkość", "Wzrost", "Waga", "Wiek", "Zwinność", "Wartość", "Pensja"])
plt.title("Heatmap")
plt.savefig("heatmap.png", bbox_inches="tight")
plt.close()


#Regresja liniowa
df_strikers = df_players[(df_players["player_positions"] == "CB")]
sns.regplot(data=df_strikers, x="defending", y="overall", line_kws={"color": "red"})

plt.title("Regresja liniowa dla środkowych obrońców")
plt.xlabel("Umiejętności obronne")
plt.ylabel("Ocena ogólna")
plt.savefig("regresja.png", bbox_inches="tight")
plt.close()


#PCA
features = ["overall", "pace", "wage_eur", "potential"]
df_pca = df_players[
    (df_players["club_name"].isin(["Real Madrid CF", "FC Barcelona", "Manchester City", "Manchester United", "Legia Warszawa"])) 
    & (df_players["player_positions"] != "GK")
    & (df_players["club_position"] != "RES")]
 
X = df_pca[features]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df_pca["PCA1"] = X_pca[:, 0]
df_pca["PCA2"] = X_pca[:, 1]

sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="club_name")
plt.title("PCA względem: ocena ogólna, szybkość, wypłata, potencjał")
plt.legend().set_title("Klub")
plt.savefig("pca.png", bbox_inches="tight")
plt.close()



