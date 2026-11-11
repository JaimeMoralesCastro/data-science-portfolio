library(tidyverse)

resultados <- data.frame(
  Algoritmo = c(
    rep("RNA", 3),
    rep("SVM", 3),
    rep("DoME", 3),
    rep("Árbol", 3),
    rep("k-NN", 3)
  ),
  Precision = c(
    0.8206, 0.8368, 0.8347,
    0.7964, 0.7822, 0.8535,
    0.7448, 0.7070, 0.8351,
    0.7438, 0.7107, 0.7107,
    0.8604, 0.8416, 0.7817
  ),
  Recall = c(
    0.7040, 0.7137, 0.6276,
    0.5561, 0.5361, 0.7836,
    0.7324, 0.7428, 0.7636,
    0.6683, 0.7107, 0.7107,
    0.7099, 0.7428, 0.6506
  ),
  F1 = c(
    0.7483, 0.7608, 0.6903,
    0.6458, 0.6268, 0.8128,
    0.7357, 0.7212, 0.7909,
    0.6935, 0.7070, 0.7070,
    0.7718, 0.7876, 0.7088
  )
)

media_metrica <- resultados %>%
  group_by(Algoritmo) %>%
  summarise(
    Precisión = mean(Precision),
    Recall = mean(Recall),
    F1 = mean(F1)
  ) %>%
  pivot_longer(cols = c(Precisión, Recall, F1),
               names_to = "Métrica",
               values_to = "Valor")

# Gráfico medias de cada algoritmo
grafico <- ggplot(media_metrica, aes(x = Métrica, y = Valor, group = Algoritmo, color = Algoritmo)) +
  geom_line(size = 1.5) +
  geom_point(size = 3) +
  labs(
    title = "Comparación de métricas por algoritmo",
    x = "Métrica",
    y = "Valor medio"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 13),  
    axis.title.x = element_text(size = 11),
    axis.title.y = element_text(size = 11)
  )

grafico

# Gráfico mejores de cada algoritmo

media_metrica <- data.frame(
  Algoritmo = rep(c("Árbol", "DoME", "k-NN", "RNA", "SVM"), each = 3),
  Métrica = rep(c("F1", "Precisión", "Sensibilidad"), times = 5),
  Valor = c(
    0.6935, 0.726, 0.711,     # Árbol
    0.791, 0.807, 0.764,      # DoME
    0.7718, 0.813, 0.743,     # k-NN
    0.761, 0.789, 0.714,      # RNA
    0.813, 0.831, 0.784       # SVM
  )
)

ggplot(media_metrica, aes(x = Algoritmo, y = Valor, fill = Métrica)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  labs(
    title = "Comparación de modelos por métrica",
    x = "Algoritmo",
    y = "Valor medio"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
    axis.title.x = element_text(size = 10),
    axis.title.y = element_text(size = 10),
    legend.title = element_blank()
  )

