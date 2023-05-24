# Fintech

Politecnico di Milano, A.Y. 2021/2022.

## Course

In [`course`](./course/) there is material and notes about the course.

## Project

Title: **Estimating Clients' Needs using Machine Learning**

Project supervisor: [Raffaele Zenti](https://www.linkedin.com/in/raffaelezenti).

Team members

- Teo Bucci ([@teobucci](https://www.github.com/teobucci))
- Filippo Cipriani ([@SmearyTundra](https://www.github.com/SmearyTundra))
- Gabriele Corbo ([@gabrielecorbo](https://www.github.com/gabrielecorbo))
- Davide Fabroni ([@davidowicz](https://www.github.com/davidowicz))
- Marco Lucchini ([@marcolucchini](https://www.github.com/marcolucchini))

### Description

The aim of this project is to exploit machine learning techniques to create a cross-platform web-app to recommend products based on clients' need.

We have been provided with the dataset [`Needs.xls`](./project/data/Needs.xls), which contains information about clients and products.

We followed a multi-tool approach, porting models to exploit each software's development strengths. We used **MATLAB**, **R**, for prototyping and **Python** with **Streamlit** module for deployment.

The prototyping part is in the [`dev`](./project/dev/) folder while the Python part is in the [`models`](./project/models/) folder; finally the [`app.py`](./project/app.py) file contains the Streamlit implementation.

### Output

The web-app is currently unavailable, but can be locally started with

```python
streamlit run project/app.py
```

Check out the final [`presentation.pdf`](./project/output/presentation.pdf).

The final evaluation can be accessed [here](./project/_ML%20Group%20Teo%20Bucci%2C%20Filippo%20Cipriani%2C%20Gabriele%20Corbo%2C%20Davide%20Fabroni%2C%20Marco%20Lucchini.pdf).
