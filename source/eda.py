import matplotlib.pyplot as plt
import pandas as pd

from log import logger

full = pd.read_csv("../data/campeonato-brasileiro-full.csv")

logger.info("Analyzing dataset")
print(full.head())
print(full.info())
full_cols = full.columns.to_list()

# Here we can already observe that the 'data' and 'hora' columns
# are of object type when they should be of datetime type

# Also the columns 'formacao_mandante', 'formacao_visitante',
# 'tecnico_mandante' and 'tecnico_visitante' have null values

# Let's check for duplicate rows

print(full.duplicated().sum())
print(full.duplicated(subset="ID").sum())

# There are no duplicate rows nor there are duplicate values for "ID"
# Now, let's analyze each feature in detail

logger.info("Analyzing each feature")
logger.info("Analyzing 'ID' column")

print(full["ID"].head(20))

# 'ID' is a pretty simple column with no duplicate values that represents
# the ID value of the match

logger.info("Analyzing 'rodata' column")

print(full["rodata"].sample(20))
print(full["rodata"].value_counts().sort_index())

# There are some rounds that appear less times than others
# A plot of the rounds may help us understand what we are dealing with

rounds_values = full["rodata"].value_counts().sort_index()
rounds = rounds_values.index
fig, ax = plt.subplots(figsize=(6, 9))
ax.barh(rounds, rounds_values)
ax.set_yticks(ticks=rounds, labels=rounds, fontsize="x-small")
ax.set_title("Rodadas jogadas")
ax.set_ylabel("Rodada")
ax.set_xlabel("Jogos")
ax.invert_yaxis()
plt.show()

# We can clearly see that most years the league had 38 rounds and only in a few
# of them it had from 39 to 46
# Let's see if the 'data' column can help us understand why

logger.info("Analyzing 'data' column")

print(full["data"])

# There are matches from 2003 to 2022
# Let's analyze the distribution of matches over the years

data = pd.to_datetime(full["data"], format="%d/%m/%Y")
years = range(2003, 2023)
matches = data.dt.year.value_counts().sort_index()
plt.bar(years, matches)
plt.xticks(ticks=years, labels=years, rotation=45, fontsize="x-small")
plt.title("Quantidade de jogos ao longo dos anos")
plt.ylabel("Quantidade de jogos")
plt.xlabel("Ano")
plt.show()

# From the plot we can observe that in 2003, 2004 and 2005 there were more
# matches than any other year. That is because back then the league had more
# rounds than nowadays. With respect to 2020 and 2021, due to the pandemic,
# some of the matches of 2020 were only played in 2021. Also, in 2016, there
# was one match less than the other years. This might be due to flaws in the
# way the data was collected.

# Let's now take a look at the 'hora' column

logger.info("Analyzing 'hora' column")

print(full["hora"])
print(full["hora"].value_counts().sort_index().index.to_list())

# There were matches basically at every shift of the day. This can be useful
# when we perform feature engineering in the future

hora = pd.to_datetime(full["hora"], format="%H:%M").dt.hour
ticks = range(0, 24, 4)
plt.hist(hora, bins=20)
plt.title("Quantidade de jogos por hora do dia")
plt.xlabel("Hora do dia")
plt.ylabel("Quantidade de jogos")
plt.xticks(ticks=ticks, label=ticks)
plt.show()

# From the graph we can see that only a few matches happened in the morning
# while most of them took place late afternoon and at night
# Let's now take a look at the 'mandante' and 'visitante' columns

logger.info("Analyzing 'mandante' column")

print(full["mandante"])
print(full["mandante"].unique().shape[0])
print(full["mandante"].value_counts().index.to_list())

# 45 differents teams played home

logger.info("Analyzing 'visitante' column")

print(full["visitante"])
print(full["visitante"].unique().shape[0])
print(full["visitante"].value_counts().index.to_list())

# 45 differents teams played away

# Just to make sure the data is consistent, let's check whether we see the same
# teams in 'mandante' and 'visitante' columns

statement = set(full["mandante"].unique()) == set(full["visitante"].unique())
print("Sets are equal") if statement else print("Sets are not equal")

# Both 'mandante' and 'visitante' have the same teams!

# Let's take a look at the 'formacao_mandante' and 'formacao_visitante' columns

logger.info("Analyzing 'formacao_mandante' column")

print(full["formacao_mandante"])
print(full["formacao_mandante"].value_counts(dropna=False, normalize=True))

logger.info("Analyzing 'formacao_visitante' column")

print(full["formacao_visitante"])
print(full["formacao_visitante"].value_counts(dropna=False, normalize=True))

# Over 60% of the rows in both 'formacao_mandante' and 'formacao_visitante' are
# null. Furthermore, this kind of information might not be useful for
# prediction since most times the team formation is only released minutes prior
# to the game. Hence, these columns are not likely to help in the prediction

# Let's take a look at the 'tecnico_mandante' and 'tecnico_visitante' columns

logger.info("Analyzing 'tecnico_mandante' column")

print(full["tecnico_mandante"])
print(full["tecnico_mandante"].value_counts(dropna=False, normalize=True))

logger.info("Analyzing 'tecnico_visitante' column")

print(full["tecnico_visitante"])
print(full["tecnico_visitante"].value_counts(dropna=False, normalize=True))

# Since almost 60% of the rows in both 'tecnico_mandante' and
# 'tecnico_visitante' are null, these columns might not be
# useful for our analysis.

# Let's take a look at the 'vencedor' column

logger.info("Analyzing 'vencedor' column")

print(full["vencedor"])
print(full["vencedor"].value_counts(normalize=True))

# The values '-' represent draws. They comprise more than 26% of all results
# Let's visualize a graph to better understand the data

winner_values = (
    full["vencedor"].str.replace(r"\B-\B", "Empate", regex=True).value_counts()
)
winners = winner_values.index
fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(winners, winner_values)
ax.set_yticks(ticks=winners, labels=winners, fontsize="x-small")
ax.set_title("Quantidade de jogos vencidos por cada time")
ax.set_ylabel("Time")
ax.set_xlabel("Quantidade de jogos")
ax.invert_yaxis()
plt.show()

# Now, let's take a look at the 'arena' column

logger.info("Analyzing 'arena' column")

print(full["arena"])
print(full["arena"].value_counts(normalize=True))

# We can see there is a weird character at the beginning of some values
# Let's see what it is
full["arena"].loc[0]

# It looks like it is a special character '\xa0'. This means that if we use
# this column in our prediction, we must clean these characters from each
# value they appear in

# Let's take a look at the 'mandante_Placar' and 'visitante_Placar' columns

logger.info("Analyzing 'mandante_Placar' column")

print(full["mandante_Placar"])
print(full["mandante_Placar"].value_counts(normalize=True).sort_index())

logger.info("Analyzing 'visitante_Placar' column")

print(full["visitante_Placar"])
print(full["visitante_Placar"].value_counts(normalize=True).sort_index())

# The data suggests that it is more likely to score more goals playing home
# compared to away. Which is to be expected from real experience
# A graph may help visualize this behavior

mandante_placar_values = full["mandante_Placar"].value_counts().sort_index()
mandante_placar = mandante_placar_values.index
visitante_placar_values = full["visitante_Placar"].value_counts().sort_index()
visitante_placar = visitante_placar_values.index

fig, ax = plt.subplots(figsize=(6, 6))
ax.barh(mandante_placar, -mandante_placar_values)
ax.barh(visitante_placar, visitante_placar_values, color="red")

ax.invert_yaxis()

ax.set_xticks(
    range(-3000, 3001, 1000), labels=[3000, 2000, 1000, 0, 1000, 2000, 3000]
)

# removing axes (spines)
for location in ["left", "right", "top", "bottom"]:
    ax.spines[location].set_visible(False)

# background box
ax.axvline(x=0, color="black", linewidth=2)

# rectangle
ax.axvspan(xmin=-3000, xmax=3000, ymin=0, ymax=1, alpha=0.3, color="grey")

# subtitles
ax.text(0.2, 1.01, "Mandante", size=10, transform=ax.transAxes)
ax.text(0.65, 1.01, "Visitante", size=10, transform=ax.transAxes)

# title and lables
ax.text(
    0.05,
    1.06,
    "Gols de Mandante vs Visitante",
    size=20,
    transform=ax.transAxes,
)
ax.set_xlabel("Quantidade de jogos em que os gols ocorreram")
ax.set_ylabel("Quantidade de gols no jogo")

plt.show()

# Let's take a look at the 'mandante_Estado' and 'visitante_Estado' columns

logger.info("Analyzing 'mandante_Estado' and 'visitante_Estado' columns")

print(full["mandante_Estado"])
print(full["mandante_Estado"].value_counts(normalize=True))

print(full["visitante_Estado"])
print(full["visitante_Estado"].value_counts(normalize=True))

# The state of the team may have a correlation with its chances of victory
# so this column can be useful in the future
