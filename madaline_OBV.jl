# Função para inicializar pesos aleatórios para as camadas
function inicializar_pesos(n_inputs, n_hidden, n_output)
    W_hidden = randn(n_hidden, n_inputs)  # Pesos da camada oculta
    W_output = randn(n_output, n_hidden)  # Pesos da camada de saída
    return W_hidden, W_output
end

# Função de ativação Threshold
function threshold(x)
    return map(xi -> xi >= 0 ? 1 : 0, x)  # Retorna 0 ou 1
end

# Função Softmax para a camada de saída
function softmax(x)
    exp_x = exp.(x .- maximum(x))  # Estabiliza o cálculo
    return exp_x ./ sum(exp_x)
end

# Função de forward pass
function forward_pass(X, W_hidden, W_output)
    X_col = reshape(X, :, 1)  # Transforma X em uma coluna
    hidden_output = threshold(W_hidden * X_col)  # Saída da camada oculta
    final_output = softmax(W_output * hidden_output)  # Saída final
    return hidden_output, final_output
end

# Função para converter a saída em vetor binário
function converter_saida(final_output)
    pred = zeros(Int64, length(final_output))
    index = argmax(final_output)  # Índice da classe com maior probabilidade
    pred[index] = 1  # Marca a classe prevista
    return pred
end

# Função para atualizar pesos
function atualizar_pesos!(X, Y_target, hidden_output, W_hidden, W_output, taxa_aprendizado)
    _, Y_pred = forward_pass(X, W_hidden, W_output)  # Forward pass
    erro = Y_target .- Y_pred  # Calcula o erro

    if any(erro .!= 0)
        W_output .+= taxa_aprendizado * erro * hidden_output'  # Atualiza pesos de saída
        for i in 1:length(hidden_output)
            if hidden_output[i] * erro[1] > 0
                W_hidden[i, :] .+= taxa_aprendizado * erro[1] * X'  # Atualiza pesos da camada oculta
            end
        end
    end
end

# Função de treinamento
function treinar(X, Y, n_hidden, epochs, taxa_aprendizado)
    W_hidden, W_output = inicializar_pesos(size(X, 2), n_hidden, size(Y, 2))  # Inicializa os pesos
    
    for epoch in 1:epochs
        for i in 1:size(X, 1)
            hidden_output, _ = forward_pass(X[i, :]', W_hidden, W_output)
            atualizar_pesos!(X[i, :]', Y[i, :], hidden_output, W_hidden, W_output, taxa_aprendizado)
        end
    end
    return W_hidden, W_output
end

# Função para testar o modelo com dados de teste
function testar_modelo(X_test, Y_test, W_hidden, W_output, obv_test)
    println("\nResultados dos testes:")
    letras = ['A', 'B', 'C', 'D', 'E']  # Mapeamento de letras
    for i in 1:size(X_test, 1)  # Loop por cada amostra de teste
        _, Y_pred_prob = forward_pass(X_test[i, :]', W_hidden, W_output)
        
        # Converte a saída para uma letra prevista
        Y_pred = converter_saida(Y_pred_prob)
        index_pred = argmax(Y_pred)
        letra_pred = letras[index_pred]
        
        # Converte a saída esperada para a letra correspondente
        index_expected = argmax(Y_test[i, :])
        letra_esperada = letras[index_expected]
        
        println("Entrada: ", X_test[i, :], 
                "\nOBV: ", obv_test[i],
                "\nPrevisão: ", letra_pred, 
                " - Esperado: ", letra_esperada)
        println("---------------------------")
    end
end


# Função para criar os dados das letras
function criar_dados_letras()
    inputs_train = [
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],  # A
        [1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0],  # B
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],  # C
        [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0],  # D
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]   # E
    ]

    targets_train = [
        [1, 0, 0, 0, 0],  # A
        [0, 1, 0, 0, 0],  # B
        [0, 0, 1, 0, 0],  # C
        [0, 0, 0, 1, 0],  # D
        [0, 0, 0, 0, 1]   # E
    ]
    
    # Dados de teste
    inputs_test = [
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],  # A (variação)
        [1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0],  # B (variação)
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],  # C (variação)
        [1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],  # D (variação)
        [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1]   # E (variação)
    ]

    targets_test = [
        [1, 0, 0, 0, 0],  # A
        [0, 1, 0, 0, 0],  # B
        [0, 0, 1, 0, 0],  # C
        [0, 0, 0, 1, 0],  # D
        [0, 0, 0, 0, 1]   # E
    ]

    X_train = hcat(inputs_train...)'
    Y_train = hcat(targets_train...)'
    X_test = hcat(inputs_test...)'
    Y_test = hcat(targets_test...)'
    return X_train, Y_train, X_test, Y_test
end

# Função para calcular OBV
function calcular_OBV(valores, volumes)
    n = length(valores)
    obv = zeros(n)
    for i in 2:n
        if valores[i] > valores[i-1]
            obv[i] = obv[i-1] + volumes[i]
        elseif valores[i] < valores[i-1]
            obv[i] = obv[i-1] - volumes[i]
        else
            obv[i] = obv[i-1]
        end
    end
    return obv
end

# Função para adicionar OBV aos dados de entrada
function adicionar_OBV(X, valores, volumes)
    obv = calcular_OBV(valores, volumes)
    obv = obv[1:size(X, 1)]  # Garante que o OBV tenha o mesmo tamanho das amostras
    X_com_OBV = hcat(X, obv)  # Adiciona o OBV como uma nova coluna
    return X_com_OBV
end

# Função principal
function main()
    input_size = 25  # 5x5 pixels
    hidden_size = 100  # Neurônios na camada oculta
    output_size = 5  # Saídas correspondentes a A, B, C, D, E

    valores = [10, 12, 11, 13, 14, 13, 15, 16, 17, 16]  # Exemplo
    volumes = [1000, 1200, 1100, 1300, 1400, 1350, 1500, 1600, 1700, 1650]

    # Cria dados de treinamento e teste
    X_train, Y_train, X_test, Y_test = criar_dados_letras()

    # Calcula OBV
    obv = calcular_OBV(valores, volumes)

    # Adiciona OBV aos dados
    X_train = adicionar_OBV(X_train, valores, volumes)
    X_test = adicionar_OBV(X_test, valores, volumes)

    # Parâmetros do treinamento
    taxa_aprendizado = 0.001
    epochs = 10000

    println("Início do treinamento...")
    W_hidden, W_output = treinar(X_train, Y_train, hidden_size, epochs, taxa_aprendizado)

    println("\nTeste do modelo com as letras treinadas:")
    testar_modelo(X_test, Y_test, W_hidden, W_output, obv[1:size(X_test, 1)])
end


main()
