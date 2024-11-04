# Projeto Madaline em Julia

Este projeto implementa uma rede neural Madaline para reconhecimento de letras (A, B, C, D, E) usando matrizes 5x5. A rede é treinada para reconhecer padrões em dados binários que representam essas letras.

## Configuração do Ambiente

Para executar este projeto, você precisará configurar a linguagem Julia no Visual Studio Code. Siga os passos abaixo:

### Passo 1: Instalar Julia

1. Baixe a versão mais recente do Julia em [julialang.org/downloads](https://julialang.org/downloads/).
2. Siga as instruções de instalação para o seu sistema operacional.

### Passo 2: Instalar o Visual Studio Code

1. Baixe o Visual Studio Code em [code.visualstudio.com](https://code.visualstudio.com/).
2. Instale o Visual Studio Code conforme as instruções fornecidas.

### Passo 3: Adicionar a Extensão Julia

1. Abra o Visual Studio Code.
2. Vá até a aba de extensões (ícone de quadrado na barra lateral esquerda) ou pressione `Ctrl + Shift + X`.
3. Pesquise por "Julia" na barra de pesquisa.
4. Selecione a extensão "Julia" e clique em "Install".

## Descrição do Código

O código implementa uma rede neural Madaline com as seguintes funcionalidades:

- **Inicialização de Pesos**: Inicializa pesos aleatórios para as camadas oculta e de saída.
- **Função de Ativação**: Utiliza uma função de ativação do tipo Threshold para a camada oculta e Softmax para a camada de saída.
- **Forward Pass**: Realiza a passagem dos dados pela rede, gerando as saídas da camada oculta e da camada de saída.
- **Treinamento**: A rede é treinada utilizando dados de entrada e saídas esperadas, atualizando os pesos com base no erro.
- **Teste do Modelo**: Avalia o modelo com um conjunto de dados de teste e compara as previsões com as saídas esperadas.

## Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu_usuario/seu_repositorio.git
   cd seu_repositorio
   
2. Abra o projeto no Visual Studio Code.

3. Execute o código no terminal do Visual Studio Code.

4. Você verá a saída das previsões do modelo com base nos dados de teste.

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

Mencionando @JuliaLang pela exelente linguagem

https://github.com/JuliaLang/julia
