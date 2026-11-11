# Portfolio de Ciencia de Datos

Cuatro proyectos independientes de aprendizaje automático, procesamiento de
señales y simulación de redes, construidos entre 2024 y 2026 durante el
grado en **Ciencia e Ingeniería de Datos** (Universidade da Coruña) y el
Erasmus+ en la **Universidad Politécnica de Bucarest**.

Cada carpeta es un proyecto completo con su código, sus resultados y su memoria.
Los datasets originales no están subidos: cada proyecto documenta de dónde
bajarlos y cómo reproducir la ejecución.

🇬🇧 [English version](README.md)

---

## Proyectos

| Proyecto | Qué responde | Stack |
|---|---|---|
| [**¿Compro o espero?**](buy-now-or-wait/) | Dada una búsqueda de vuelo, ¿bajará el precio al menos un 5 % en una semana? Cuatro modelos comparados sobre 6,1 M de búsquedas reales de Expedia, con el umbral de decisión elegido por dólares ahorrados y no por F1. | PySpark · MLlib · GBT · Parquet |
| [**Clasificación de minas con sonar**](sonar-mine-classification/) | ¿Mina o roca? Cinco familias de modelos comparadas bajo un criterio de riesgo asimétrico, donde no detectar una mina cuesta mucho más que una falsa alarma. | Julia · SVM · RNA · k-NN · DoME |
| [**Detección de lenguaje abusivo**](abusive-language-detection/) | ¿Aguanta un modelo de toxicidad entrenado con Wikipedia cuando le llega Twitter? Un diseño 3 × 3 de entrenamiento y evaluación responde que no del todo. | PyTorch · RoBERTa · Hugging Face |
| [**Simulación de propagación de rumores**](rumor-spread-sir-simulation/) | ¿Cómo cambia la forma de la red la velocidad de un rumor? SIR basado en agentes sobre grafos libres de escala y de mundo pequeño. | Python · NetworkX · SIR |

---

## Lo que comparten estos proyectos

- **La decisión va antes que la métrica.** En el modelo de vuelos el punto de
  trabajo se elige por ahorro esperado; en el de minas, por sensibilidad,
  porque un falso negativo es una mina en el fondo del mar. La accuracy casi
  nunca es la magnitud que importa.
- **Validación que se parece a la realidad.** Particiones temporales por fecha
  de vuelo en vez de aleatorias, validación cruzada cuando la muestra es
  pequeña, y conjuntos de test fuera de dominio cuando la pregunta es si
  generaliza.
- **Fugas buscadas a propósito.** El proyecto de vuelos documenta cómo la
  primera definición de la etiqueta usaba información que no se conoce al
  buscar, y cómo una ventana fija redujo ese sesgo de 23 puntos a 9.

---

## Hacia dónde va esto

Los modelos son el medio, no el objetivo. Lo que me interesa es el final de la
cadena: la **decisión que sostiene cada análisis** y lo que vale adoptarla, en
dinero, en tiempo o en riesgo.

Esa es la misma pregunta que se hace una operación todos los días. El
razonamiento que elige un umbral de comprar o esperar a partir de una matriz de
costes es el que fija un stock de seguridad, un punto de pedido o un nivel de
servicio; el criterio asimétrico que se niega a no detectar una mina es el que
se niega a no detectar una rotura de stock o una entrega fallida; y el modelo
basado en agentes que mide a qué velocidad se propaga algo por una red sirve
igual para una red de distribución que para una red social.

Así que la dirección en la que trabajo es la **logística empresarial e
industrial** —operaciones, cadena de suministro y planificación—, donde lo que
se entrega no es una métrica sobre un conjunto de test, sino una regla de
operación que alguien pueda seguir, con el coste de equivocarse escrito al
lado.

---

## Estructura

```
buy-now-or-wait/              pipeline de Spark, notebooks, métricas, memoria
sonar-mine-classification/    scripts de Julia, dataset Sonar, matrices de confusión
abusive-language-detection/   entrenamiento y evaluación de RoBERTa, resultados 3x3
rumor-spread-sir-simulation/  modelo basado en agentes, gráficas, memoria
```

Cada proyecto tiene su propio `README.md` con requisitos, cómo conseguir los
datos y cómo ejecutarlo.

---

## Autor

**Jaime Morales Castro** — Ciencia e Ingeniería de Datos, Universidade da
Coruña · Erasmus+ en la Universidad Politécnica de Bucarest.
[LinkedIn](https://linkedin.com/in/jaime-morales-castro)

Los trabajos académicos acreditan a sus coautores en el README de cada proyecto.

## Licencia

El código se publica bajo [licencia MIT](LICENSE). Las memorias, presentaciones
y datasets de terceros mantienen sus propios términos; los datasets
referenciados (Expedia Flight Prices, UCI Sonar, Jigsaw, OLID) **no** se
redistribuyen aquí.
