<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<a name="readme-top"></a>

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/matheusbitaraes/AIHero">
    <img src="documents/readme/logo_white_no_bg.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">AI Hero Server</h3>

  <p align="center">
    An Artificial Intelligence built for generating musical melodies
    <br />
    <br/>
    <a href="https://github.com/matheusbitaraes/AIHero">
        <img src="documents/readme/gan_training.gif" alt="training animation" width="90%" height="30%">
    </a>
    <br />
    <br />
    <a href="https://aihero.bitaraes.com.br">View Demo</a>
    ·
    <a href="https://github.com/matheusbitaraes/aihero/issues">Report Bug</a>
    ·
    <a href="https://github.com/matheusbitaraes/aihero/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

The AI Hero project proposes a blues melody generator, that tries to emulate the improvisation process of the human mind.

![Project Architecture][project-architecure]

This work proposes an architecture composed
of a genetic algorithm whose initial population is fed by generative adversarial networks
(GANs) specialized in generating melodies for certain harmonic functions. The fitness
function of the genetic algorithm is a weighted sum of heuristic methods for evaluating
quality, where the weights of each function are assigned by the user, before requesting
the melody. A data augmentation statregy for the GAN training data was proposed and
experimentally validated. This experiment and two others are available in the [masters thesis](https://www.ppgee.ufmg.br/defesas/2006M.PDF) (in portuguese) generated by this work.

Also, [this article](https://www.sba.org.br/cba2022/wp-content/uploads/artigos_cba2022/paper_1817.pdf), validating a data augmentation strategy proposal, was published as a consequence of the work.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

- [![React][react.js]][react-url]
- [![Vue][vue.js]][vue-url]
- [![Angular][angular.io]][angular-url]
- [![Svelte][svelte.dev]][svelte-url]
- [![Laravel][laravel.com]][laravel-url]
- [![Bootstrap][bootstrap.com]][bootstrap-url]
- [![Python][python.com]][python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->

<!-- GETTING STARTED -->

## Getting Started

Below is the guideline on how to run the python server locally for training or generating melodies

### Installation

This project was developed using `Python 3.8`. So, make sure you are using a compatible version.
Then, install the dependencies.

- Install dependencies

  ```sh
  pip install -r requirements.txt
  ```

- Install dependencies for mac - A different tensorflow needs to be used
  ```sh
  pip install -r requirements-mac.txt
  ```

<!-- _Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = "ENTER YOUR API";
   ``` -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Model Training

To train the Model (GAN), you should go to the `src/GEN` folder and run the `train_script.py`. All important configurations are available in `src/config.json`

<!-- _For more examples, please refer to the [Documentation](https://example.com)_ -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Model Execution

To execute the Model as a server, you should run

```sh
python src/main.py src/config.json
```

Then, the server, with loaded models, will be available at port `8083` (this can be modified in the `src/config.json`).

## Requesting a Melody

To actually request a melody, you could clone the [front-end project](https://github.com/matheusbitaraes/AIHeroFront) and execute it:
[![Product Name Screen Shot][product-screenshot]](https://aihero.bitaraes.com.br)

Alternativelly you could make the REST requests by yourself. Here is how to do it:

Initially you should ask for the melody using a `POST` request, as follows:

url:

```sh
http://localhost:8083/melody?source=${source}
```

where `source` can have one of following values: `train`, `gan` or `evo`

body:

```yml
{
  harmony_specs:
    [
      { melodic_part: "", chord: "C:7maj", key: "C", tempo: 120 },
      { melodic_part: "", chord: "F:7maj", key: "C", tempo: 120 },
      { melodic_part: "", chord: "C:7maj", key: "C", tempo: 120 },
      { melodic_part: "", chord: "C:7maj", key: "C", tempo: 120 },
      { melodic_part: "", chord: "F:7maj", key: "C", tempo: 120 },
      { melodic_part: "", chord: "F:7maj", key: "C", tempo: 120 },
      { melodic_part: "", chord: "C:7maj", key: "C", tempo: 120 },
      { melodic_part: "", chord: "C:7maj", key: "C", tempo: 120 },
      { melodic_part: "", chord: "G:7maj", key: "C", tempo: 120 },
      { melodic_part: "", chord: "F:7maj", key: "C", tempo: 120 },
      { melodic_part: "", chord: "C:7maj", key: "C", tempo: 120 },
      { melodic_part: "", chord: "G:7maj", key: "C", tempo: 120 },
    ],
  evolutionary_specs:
    [
      {
        key: "notes_on_same_chord_key",
        name: "Notes on Same Chord",
        description: "notes_on_same_chord_key",
        transf_weights: [Array],
        bounds: [Array],
        weight: 0,
      },
      {
        key: "notes_on_beat_rate",
        name: "Notes on Beat",
        description: "notes_on_beat_rate",
        transf_weights: [Array],
        bounds: [Array],
        weight: 0,
      },
      {
        key: "note_on_density",
        name: "Note Density",
        description: "note_on_density",
        transf_weights: [Array],
        bounds: [Array],
        weight: 1,
      },
      {
        key: "note_variety_rate",
        name: "Note Variety",
        description: "note_variety_rate",
        transf_weights: [Array],
        bounds: [Array],
        weight: 0,
      },
      {
        key: "single_notes_rate",
        name: "Single Notes Rate",
        description: "single_notes_rate",
        transf_weights: [Array],
        bounds: [Array],
        weight: 1,
      },
      {
        key: "notes_out_of_scale_rate",
        name: "Notes out of Scale",
        description: "notes_out_of_scale_rate",
        transf_weights: [Array],
        bounds: [Array],
        weight: 0,
      },
    ],
}
```

You will then receive the `melodyId` in the response, which will be an MD5 hash (ex: `205f1c3b-da87-455f-94b6-a8f49cb346b9`).

Then, you can query for the melody by making a `GET` request with

```sh
localhost:8083/melody/${melodyId}
```

If the melody is not ready, you will receive a response `404`, if it is ready, you will receive a response with status `200` and the .mid file.

<!-- _For more examples, please refer to the [Documentation](https://example.com)_ -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->

## Contact

For more clarification, please reach out!

Matheus Bitarães - [LinkedIn](https://linkedin.com/in/matheus-bitaraes) - matheusbitaraesdenovaes@gmail.com

Project Link: [https://aihero.bitaraes.com.br](https://aihero.bitaraes.com.br)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- Template created based on [this template](https://github.com/othneildrew/Best-README-Template/blob/master/README.md)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/matheus-bitaraes
[product-screenshot]: documents/readme/GUI.png
[project-architecure]: documents/readme/architecture.png
[next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[next-url]: https://nextjs.org/
[react.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[react-url]: https://reactjs.org/
[vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[vue-url]: https://vuejs.org/
[angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[angular-url]: https://angular.io/
[svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[svelte-url]: https://svelte.dev/
[laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
