# -*- coding: utf-8 -*-TÍTULO
"""
CLASSIFICADOR DE DESCRIÇÕES DE ITENS EM DOCUMENTOS FISCAIS DE VENDA DE COMBUSTÍVEL

- Baseado no modelo de classificador com CountVectorizer
- Dados de entradas: arquivo excel 'Datasets.xlsx', com três abas
- "Descrições de itens" aqui são chamadas de "sentenças"

CONTROLE DE VERSÕES
    - 2.8 - Limpeza geral para a apresentação do TCC; melhoria no tamanho da fonte das figuras de "confusion matrix"
        
"""

# %% - MÉTODOS GERAIS
"""
Bibliotecas e métodos que serão utilizadas em diversos blocos de código
"""

# Bibliotecas
import time                       # Medição de tempos de processamento: vetorização, treinamento...
import sys                        # Medição de taxas de processamento
import matplotlib.pyplot as plt   # Figuras
import seaborn as sns             # Figuras
import pandas as pd               # Tratamento de dados
from sklearn.metrics import confusion_matrix     # Geração de matriz de confusão
from sklearn.metrics import balanced_accuracy_score    # Métrica importante, a balanced_accuracy
from sklearn.metrics import classification_report  # Geração de relatório de avaliação
import re                         # Biblioteca para expressão Regulares

# Inicialização de variáveis de contagem de tempos
started_at = time.time()
time_consumed = time.time()
bloco = ''

# Métodos de contagem de tempo
def start_time_counter(bloco):
    print('-------------------------------------------------------------------')
    print('Inicio de ' + bloco + '...')
    global started_at
    started_at = time.time()
    
def end_time_counter(bloco):
    global time_consumed
    time_consumed = time.time() - started_at
    print(bloco + f' finalizado. Tempo consumido: { round((time_consumed), 6)} segundos.')
    
def end_time_rate_counter(bloco, variavel):
    global time_consumed
    time_consumed = time.time() - started_at
    tamanhoMB = sys.getsizeof(variavel)/(1024*1024)
    print(f'Tamanho da variável tratada: {tamanhoMB:.2f} MB')
    try:
        print(f'Velocidade de processamento: {(tamanhoMB/time_consumed):.2f} MB/s')
    except ZeroDivisionError:
        pass

# %% - IMPORTAÇÃO DOS DADOS
"""
IMPORTAÇÃO DE DADOS
    - Será criado um dataframe DF com os dados importados e rotulados
    - A origem é um arquivo excel disponível localmente
"""

bloco = 'IMPORTAÇÃO DE DADOS'
start_time_counter(bloco)

# O arquivo Datasets.xlsx deve ter três planilhas:
# 'RecorteGeografico'       # Dataset A         dados de uma regional ao longo de cinco anos
# 'RecorteTemporal'         # Dataset B         dados do estado inteiro, julho/2024
# 'RecorteTemporalUN'       # Dataset B_UN      dados do estado inteiro, julho/2024, com unidade de medida
    
# Escolha do dataset a ser tratado, via comentário em linha de código:
# dataset_escolhido = 'Dataset_A'   # 'RecorteGeografico'   # Dataset_A
# dataset_escolhido = 'Dataset_B'   # 'RecorteTemporal'     # Dataset_B
dataset_escolhido = 'Dataset_B_UN'  # 'RecorteTemporalUN'       # Dataset_B_UN

# Importação via pandas, especificando que tudo deve ser TEXTO (str)
# Poderá aparecer um warning sobre a validação de dados, que pode ser ignorado
df = pd.read_excel(    
    'Datasets.xlsx',
    sheet_name=dataset_escolhido,
    header=0,
    index_col=None,
    dtype={'DS_ITEM': str, 'DS_MARCADOR_FISCAL': str, 'VL_TOTAL': float} )

# Verifica se havia a coluna "unidade de medida" no dataset e adiciona-a à descrição de item
unidade_medida_presente = 'CD_UNID_MEDIDA' in df.columns 
if unidade_medida_presente:
    df['CD_UNID_MEDIDA'] = df['CD_UNID_MEDIDA'].astype(str)    # Ajustar o tipo de dados da coluna 'CD_UNID_MEDIDA' para string
    df['DS_ITEM'] = df['DS_ITEM'] + ' ' + df['CD_UNID_MEDIDA']
    df.drop(columns=['CD_UNID_MEDIDA'], inplace=True)
    
# Informações finais
end_time_counter(bloco)

# %% - VISUALIZAÇÃO INICIAL DO DATASET
"""
VISUALIZAÇÃO INICIAL DOS DADOS
    - Criação de variáveis:
        SENTENCAS
        ROTULOS
        VOLUME_VENDIDO
"""

bloco = 'VISUALIZAÇÃO PRÉVIA DE DADOS'
start_time_counter(bloco)

# Informações úteis sobre o dataset
print("\n\nInformações sobre o dataset:")
print(df.info())

# Informações sobre o conjunto de dados
print("\n\nInformações sobre os valores dos dados:")
print(df.describe(include='all'))

# As 5 primeiras linhas do conjunto de dados
print("\nPrimeiras sentenças:")
print(df.head())

# As 5 últimas linhas do conjunto de dados
print("\nÚltimas sentenças:")
print(df.tail())

# Variáveis criadas para as sentenças, rótulos e pesos das amostras, evitar uso da notação df[...]
# Poderia excluir o df, mas como o espaço aqui não é crítico, foi mantido
sentencas = df['DS_ITEM']
rotulos = df['DS_MARCADOR_FISCAL']
volumes_vendidos = df['VL_TOTAL']  
if 'CD_UNID_MEDIDA' in df.columns:          # Na verdade não é necessário, mas mantido para eventual experimentação
    unidades_medida = df['CD_UNID_MEDIDA']

# Visualização do (des)balanceamento das categorias
print("\nFrequências das categorias:")      # Distribuição das classes
print(rotulos.value_counts())


# Gráfico de barras do dataset, para visualização do desbalanceamento:

# Contando os valores e criando o gráfico de barras com escala logarítmica no eixo Y
ax = rotulos.value_counts().plot(kind='bar', logy=True)
# Nomeando os eixos e adicionando um título
ax.set_xlabel('Categorias')
ax.set_ylabel('Contagem (logarítmica)')
ax.set_title('Distribuição das Categorias')
# Removendo as bordas do gráfico (padrão TCC USP)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# Adicionando os valores no topo de cada barra
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
# Exibindo o gráfico
print("\nApresentando gráfico de barras...\n")      # Distribuição das classes
plt.show()

# Informações finais
end_time_rate_counter(bloco, df)
end_time_counter(bloco)

# %% - TRATAMENTO INICIAL DOS DADOS
"""
TRATAMENTO INICIAL DOS DADOS
LIMPEZA INICIAL DAS SENTENÇAS A SEREM CLASSIFICADAS
    - No caso, há opções aqui de remoção de ESPAÇOS, CARACTERES ESPECIAIS, ETC, por meio de expressões regulares.
    - Criação de variável:
        PROCESSED_LINES (sentenças já tratadas para facilitar tokenização: lowercase, remoção de lixo...)
"""

bloco = 'LIMPEZA INICIAL DAS SENTENÇAS'
start_time_counter(bloco)
  
# Retira caracteres não imprimíveis, substituindo-os por espaços. Substituir por espaços é mais adequado do que 
# somente suprimir porque não gera efeito de concatenamento de tokens.
processed_lines = sentencas.str.lower().replace(r'[^a-zA-Z0-9\s]',' ', regex=True)

# Também, força que todas sejam strings:
processed_lines = processed_lines.astype(str)

# Tratamentos específicos:
# Resolver alguns casos de S10 e S500 que pareciam estar separados S 10 e S 500.
processed_lines = processed_lines.str.replace(r'\bs\s+10\b', 's10', regex=True)
processed_lines = processed_lines.str.replace(r'\bs\s+500\b', 's500', regex=True)
# Caso encontre a palavra 'etanol' misturada com outras, separa-a
processed_lines = processed_lines.str.replace(r'(\w*)etanol(\w*)', r'\1 etanol \2', regex=True)
# Caso encontre a palavra 'vpower' misturada com outras, separa-a
processed_lines = processed_lines.str.replace(r'(\w*)vpower(\w*)', r'\1 vpower \2', regex=True)
# Substitui a palavra isolada "adi" por "aditivado"
processed_lines = processed_lines.str.replace(r'\badi\b', 'aditivado', regex=True)

# Substituir valores numéricos isolados por 'nmbr', reduz dimensionalidade e melhora precisão
processed_lines = processed_lines.str.replace(r'\b\d+\b', 'nmbr', regex=True)

# Tratamentos finais
# Substitui dois ou mais espaços em branco por um só
processed_lines = processed_lines.str.replace(r'\s+',' ', regex=True)
# Remove espeços em branco à esquerda e à direita
processed_lines = processed_lines.str.replace(r'^\s+|\s+?$','', regex=True)

end_time_counter(bloco)

# %% - SERIALIZAÇÃO/DESSERIALIZAÇÃO DO DATASET TRATADO
"""
DEPOIS DE TANTO TRABALHO, SERIALIZAR / DESSERIALIZAR O 'CORPUS' LIMPO PARA USO SEM PRECISAR IMPORTAR E PROCESSAR NOVAMENTE
"""
import pickle

bloco = 'SERIALIZAÇÃO E DESSERIALIZAÇÃO DAS VARIÁVEIS'
start_time_counter(bloco)

# Serialização e desserialização em modo binário

# Serializando o corpus
with open('sentencas.pkl', 'wb') as file:      
    pickle.dump(processed_lines, file)
print("O CORPUS foi salvo no arquivo 'sentencas.pkl'.")

# Serializando os rótulos
with open('rotulos.pkl', 'wb') as file:      
    pickle.dump(rotulos, file)
print("Os RÓTULOS foram salvos no arquivo 'rotulos.pkl'.")

# Serializando os volumes vendidos
with open('volumes_vendidos.pkl', 'wb') as file:      
    pickle.dump(volumes_vendidos, file)
print("Os PESOS foram salvos no arquivo 'volumes_vendidos.pkl'.")

# Agora daria até para apagar as variáveis 'processed_lines' e 'classes' no Variable Explorer.
# Caso fosse feito, carregaria as variáveis com o código abaixo 

# Desserializando o corpus
with open('sentencas.pkl', 'rb') as file:
    processed_lines = pickle.load(file)
print("O CORPUS foi recuperado do arquivo 'sentencas.pkl'.")

# Desserializando os rótulos
with open('rotulos.pkl', 'rb') as file:
    rotulos = pickle.load(file)
print("Os RÓTULOS foram recuperados do arquivo 'rotulos.pkl'.")

# Desserializando os volumes_vendidos
with open('volumes_vendidos.pkl', 'rb') as file:
    volumes_vendidos = pickle.load(file)
print("Os VOLUMES VENDIDOS foram recuperados do arquivo 'volumes_vendidos.pkl'.")

end_time_rate_counter(bloco, df)
end_time_counter(bloco)

# %% - VETORIZAÇÃO
"""
VETORIZAÇÃO DO CORPUS LIMPO
    - Utilização do CountVectorizer
    - Não se descartou nenhuma feature
    - Variável criada:
        VECTORS: é o "X", uma matriz com todas as sentenças vetorizadas
"""
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

bloco = 'VETORIZAÇÃO'
start_time_counter(bloco)

# Instanciação do vetorizador
vectorizer = CountVectorizer(
    strip_accents='unicode',        #
    token_pattern=r"(?u)\b\w+\b",   # modificado para incluir palavras de um caractere
    analyzer='word',                # tokeniza em palavras
    ngram_range=(1,2)               # tokeniza em unigramas e bigramas
    )

# Caso quisesse, ao mesmo tempo, aprender o vocabulário E vetorizar, usaria o fit_transform.
# Mas como quero salvar o vocabulário em separado, separo em fit() e transform()

# FIT - Ajustando o modelo (tokenização) - "aprendendo o vocabulário" e salvando o vetorizador treinado
vectorizer.fit(processed_lines)
with open('count_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
print("O ajuste do modelo foi salvo no arquivo 'count_vectorizer.pkl'.")

# TRANSFORM - Vetorização do CORPUS
print('Vetorizando (palavras em tokens)...')
vectors = vectorizer.transform(processed_lines)
X = vectors         # Por questões de padronização, X: preditoras, y: predita

# Contagem de tempo padrão
time_consumed = time.time() - started_at
print(bloco + f' finalizado. Tempo consumido: { round((time_consumed), 6)} segundos.')
tamanhoMB = sys.getsizeof(processed_lines)/(1024*1024)
print(f'Tamanho do processed_lines: {tamanhoMB:.2f} MB')
print(f'Velocidade de processamento: {(tamanhoMB/time_consumed):.2f} MB/s')

end_time_counter(bloco)

# %% - CODIFICAÇÃO AUTOMÁTICA DOS LABELS
"""
CODIFICAÇÃO AUTOMÁTICA DOS LABELS, MODELO FICA FLEXÍVEL À VARIAÇÃO DO NÚMERO DE CATEGORIAS
    - Variável criada:
        MY_TARGETS: é o "y"
"""

bloco = 'CODIFICAÇÃO AUTOMÁTICA DE LABELS'
start_time_counter(bloco)

# Biblioteca para converter valor categórico em discreto 
from sklearn.preprocessing import LabelEncoder

# Há outros encoders, mas este basta
encoder = LabelEncoder()

print("\nCodificando os targets (Strings) para valores numéricos...")    

# rotulos_codificados_em_numeros = encoder.fit_transform(rotulos)
my_targets = encoder.fit_transform(rotulos)
y = my_targets          # Por questões de padronização. X são as preditoras, y é a predita.

# Apresenta os rótulos e sua codificação interna
print('Rótulos encontrados:')
print(encoder.classes_)

end_time_counter(bloco)

# %% - DIVISÃO EM TREINO E TESTE
"""
TRATAMENTO DO MODELO DE PREDIÇÃO - DIVISÃO EM CONJUNTOS DE TREINAMENTO E TESTE
    - sklearn: TRAIN_TEST_SPLIT
    - Variáveis criadas:
        X_train: TRAIN FEATURES
        X_test: TEST FEATURES
        y_train: TRAIN TARGETS
        y_test: TEST TARGETS

"""

bloco = 'DIVISÃO DO DATASET EM TREINO E TESTE'
start_time_counter(bloco)

# Biblioteca para dividir o dataset
from sklearn.model_selection import train_test_split

# Para EVENTUAL tratamento de uma matriz esparsa
# from scipy.sparse import csr_matrix

# Separador de milhar brasileiro
import locale
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

# Divisão dos dados em dois conjuntos; 75% treinamento e 25% teste
# Automaticamente, O TRAIN_TEST_SPLIT já faz o embaralhamento antes da divisão
# Importante indicar que a divisão dos dados deve manter a mesma proporção de classes nos conjuntos de treino e teste (stratify=y).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=28071977)

# Imprimir número de instâncias no conjunto de treinamento e no de teste
num_linhas_treino = X_train.shape[0]
num_linhas_treino_formatado = locale.format_string("%d", num_linhas_treino, grouping=True) #     f"{num_linhas_treino:,}"
print(f'Tamanho do dataset de treino:\t {num_linhas_treino_formatado} amostras.')

num_linhas_teste = X_test.shape[0]
num_linhas_teste_formatado = locale.format_string("%d", num_linhas_teste, grouping=True) #     f"{num_linhas_treino:,}"
print(f'Tamanho do dataset de teste:\t {num_linhas_teste_formatado} amostras.')

end_time_counter(bloco)

# %% - CALCULAR A COMPENSAÇÃO DO DESBALANCEAMENTO
"""
APÓS A DIVISÃO DO EM TREINO E TESTE, CALCULAR A COMPENSAÇÃO DO DESBALANCEAMENTO, 
CASO SE UTILIZE O PESO DAS AMOSTRAS (SAMPLE_WEIGHT), COLOCANDO NO DATASET DE **TREINO** OS PESOS DAS AMOSTRAS
O PROCEDIMENTO ACABOU NÃO SENDO UTILIZADO, UTILIZANDO-SE EM SEU LUGAR A MÉTRICA "BALANCED_ACCURACY" 
NO AJUSTE DE HIPERPARÂMETROS
"""
from sklearn.utils.class_weight import compute_sample_weight
my_sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)   # Primeiro, no dataset de TREINO

# %% - CLASSIFICADORES - INSTANCIAÇÃO
"""
INSTANCIAÇÃO DE CLASSIFICADORES NAÏVE BAYES E RANDOM FOREST
"""

bloco = 'INSTANCIAÇÃO DOS CLASSIFICADORES COM PARÂMETROS GENÉRICOS'
start_time_counter(bloco)

# Bibliotecas do classificador Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
# Biblioteca do classificador Random Forest
from sklearn.ensemble import RandomForestClassifier

# Instanciando os classificadores
model_mnb = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
model_cnb = ComplementNB(alpha=1.0)
model_rf = RandomForestClassifier(n_estimators=100, verbose=2, random_state=28071977)

# Armazenando os modelos em um dicionário
modelos = {
    'Classificador_MNB': model_mnb,
    'Classificador_CNB': model_cnb,
    'Classificador_RF': model_rf
}

print('\nModelos e hiperparâmetros iniciais:')
for nome_modelo, modelo in modelos.items():
    print(f'\nModelo: {nome_modelo}')
    print(type(modelo))
    params_non_optimized = modelo.get_params()
    # Imprimir os hiperparâmetros
    for param, value in params_non_optimized.items():
        print(f"{param}: {value}")
        
end_time_counter(bloco)
     
# %% - CLASSIFICADORES - TREINAMENTO INICIAL
"""
TREINAMENTO INICIAL COM HIPERPARÂMETROS "GENÉRICOS"
    - O modelo Random Forest foi deixado de fora por consumir recursos excessivamente 
"""
bloco = 'TREINAMENTO INICIAL COM HIPERPARÂMNETROS GENÉRICOS'
start_time_counter(bloco)

print('Treinando modelo MNB...')
fit_time = time.time()
model_mnb.fit(X_train, y_train, my_sample_weights)
time_consumed = time.time() - fit_time
print(f'Treinando modelo MNB finalizado. Tempo consumido (segundos): {time_consumed:.2f}')

print('Treinando modelo CNB...')
fit_time = time.time()
model_cnb.fit(X_train, y_train, my_sample_weights)
time_consumed = time.time() - fit_time
print(f'Treinando modelo CNB finalizado. Tempo consumido (segundos): {time_consumed:.2f}')

"""
print('Treinando modelo RF...')
fit_time = time.time()
model_rf.fit(X_train, y_train, my_sample_weights)
time_consumed = time.time() - fit_time
print(f'Treinando modelo RF finalizado. Tempo consumido (segundos): {time_consumed:.2f}')
"""

end_time_counter(bloco)

# %% - CLASSIFICADORES - PREDIÇÃO INICIAL ANTES DA OTIMIZAÇÃO DOS HIPERPARÂMETROS, RELATÓRIOS DE DESEMPENHO, MATRIZES DE CONFUSÃO
"""
AVALIAÇÃO (MÉTRICAS) DO MODELO DE PREDIÇÃO ***AINDA SEM OTIMIZAÇÃO***
    - Esta avaliação é feita com base em dataset de TESTE.
    - Só para efeitos de verificação de overfitting, são feitas métricas para o dataset de treino também.
"""

bloco = 'AVALIAÇÃO DO MODELO AINDA NÃO OTIMIZADO'
start_time_counter(bloco)

# Predições antes da otimização dos hiperparâmetros - dataset treino
y_pred_train_mnb_before = model_cnb.predict(X_train)
y_pred_train_cnb_before = model_mnb.predict(X_train)
# y_pred_train_rf_before = model_rf.predict(X_train)

# Predições antes da otimização dos hiperparâmetros - dataset teste
y_pred_test_mnb_before = model_cnb.predict(X_test)
y_pred_test_cnb_before = model_mnb.predict(X_test)
# y_pred_rf_before = model_rf.predict(X_test)


# Matrizes de confusão
cm_non_optimized_mnb = confusion_matrix(y_test, y_pred_test_mnb_before)
cm_non_optimized_cnb = confusion_matrix(y_test, y_pred_test_cnb_before)
# cm_non_optimized_rf = confusion_matrix(y_test, y_pred_rf_before)

# Aplicar escala logarítmica às matrizes de confusão, pois há uma categoria muito dominante, que prejudica a visualização em heatmap
cm_non_optimized_mnb_log = np.log1p(cm_non_optimized_mnb)
cm_non_optimized_cnb_log = np.log1p(cm_non_optimized_cnb)
# cm_non_optimized_rf_log = np.log1p(cm_non_optimized_rf)

# Transformar em dataframe para usar no heatmap. ANNOTATIONS (VALORES NUMÉRICOS).
df_cm_non_optimized_mnb = pd.DataFrame(
    cm_non_optimized_mnb,
    index=encoder.classes_,
    columns=encoder.classes_
)
df_cm_non_optimized_cnb = pd.DataFrame(
    cm_non_optimized_cnb,
    index=encoder.classes_,
    columns=encoder.classes_
)
"""
df_cm_non_optimized_rf = pd.DataFrame(
    cm_non_optimized_rf,
    index=encoder.classes_,
    columns=encoder.classes_
)
"""

# Transformar em dataframe para usar no heatmap. CORES.
df_cm_non_optimized_mnb_log = pd.DataFrame(
    cm_non_optimized_mnb_log,
    index=encoder.classes_,
    columns=encoder.classes_
    )
df_cm_non_optimized_cnb_log = pd.DataFrame(
    cm_non_optimized_cnb_log,
    index=encoder.classes_,
    columns=encoder.classes_
    )
"""
df_cm_non_optimized_rf_log = pd.DataFrame(
    cm_non_optimized_rf_log,
    index=encoder.classes_,
    columns=encoder.classes_
    )
"""
# Acurácias balanceadas:

balanced_accuracy_train_mnb_non_optimized = balanced_accuracy_score(y_train, y_pred_train_mnb_before)
balanced_accuracy_train_cnb_non_optimized = balanced_accuracy_score(y_train, y_pred_train_cnb_before)
# balanced_accuracy_train_rf_non_optimized = balanced_accuracy_score(y_train, y_pred_train_rf_before)

balanced_accuracy_mnb_non_optimized = balanced_accuracy_score(y_test, y_pred_test_mnb_before)
balanced_accuracy_cnb_non_optimized = balanced_accuracy_score(y_test, y_pred_test_cnb_before)
# balanced_accuracy_rf_non_optimized = balanced_accuracy_score(y_test, y_pred_test_rf_before)

# RELATÓRIOS DE CLASSIFICAÇÃO ANTES DA OTIMIZAÇÃO DOS HIPERPARÂMETROS

# MNB
print("Relatório de classificação para MNB antes da otimização:")
print(pd.DataFrame(classification_report(
        y_test,
        y_pred_test_mnb_before,
        target_names=encoder.classes_,  # Isso aqui deu o bicho para descobrir
        digits=3,
        output_dict=True,
        zero_division=np.nan            # Para não ficar dando warnings quando há zero predições
        )).transpose()
    )
print(f'Acurácia balanceada MNB - TESTE: {balanced_accuracy_mnb_non_optimized}')

# CNB
print("Relatório de classificação para CNB antes da otimização:")
print(pd.DataFrame(classification_report(
        y_test,
        y_pred_test_cnb_before,
        target_names=encoder.classes_,  # Isso aqui deu o bicho para descobrir
        digits=3,
        output_dict=True,
        zero_division=np.nan            # Para não ficar dando warnings quando há zero predições
        )).transpose()
    )
print(f'Acurácia balanceada CNB - TESTE: {balanced_accuracy_cnb_non_optimized}')

"""
# RF
print("Relatório de classificação para RF antes da otimização:")
print(pd.DataFrame(classification_report(
        y_test,
        y_pred_test_rf_before,
        target_names=encoder.classes_,  # Isso aqui deu o bicho para descobrir
        digits=3,
        output_dict=True,
        zero_division=np.nan            # Para não ficar dando warnings quando há zero predições
        )).transpose()
    )
print(f'Acurácia balanceada RF - TESTE: {balanced_accuracy_rf_non_optimized}')
"""

# MATRIZES DE CONFUSÃO ANTES DA OTIMIZAÇÃO DOS HIPERPARÂMETROS

# Tamanhos de fonte
fonte_dos_numeros = 16
fonte_do_titulo = 16
fonte_do_xylabel = 16

# MNB
print(confusion_matrix(y_test, y_pred_test_mnb_before))
plt.figure(figsize=(12,10))
sns.heatmap(df_cm_non_optimized_mnb_log.astype(int), annot=df_cm_non_optimized_mnb, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Cores em Escala Logarítmica'}, annot_kws={"size": fonte_dos_numeros}) 
plt.title(f'Dataset: {dataset_escolhido} - Conjunto de Teste - Antes da Otimização de Hiperparâmetros\n \
          Algoritmo: {type(model_mnb)}\nVetorizador: {type(vectorizer)}; N-gramas: {vectorizer.ngram_range}\n \
          Acurácia Balanceada: {balanced_accuracy_mnb_non_optimized:.4f}; Alpha: {model_mnb.alpha:.3f}\n    \
          Unidade de medida no dataset: {unidade_medida_presente}', fontsize=fonte_do_titulo)  # Aumenta o tamanho da fonte do título
plt.xlabel('Classe Predita', fontsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte do rótulo do eixo x
plt.ylabel('Classe Real', fontsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte do rótulo do eixo y
plt.tick_params(axis='x', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo x
plt.tick_params(axis='y', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo y
plt.show()

#---------------------------------------------------------------------------

# CNB
print(confusion_matrix(y_test, y_pred_test_cnb_before))
plt.figure(figsize=(12,10))
sns.heatmap(df_cm_non_optimized_cnb_log.astype(int), annot=df_cm_non_optimized_cnb, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Cores em Escala Logarítmica'}, annot_kws={"size": fonte_dos_numeros}) 
plt.title(f'Dataset: {dataset_escolhido} - Conjunto de Teste - Antes da Otimização de Hiperparâmetros\n \
          Algoritmo: {type(model_cnb)}\nVetorizador: {type(vectorizer)}; N-gramas: {vectorizer.ngram_range}\n \
          Acurácia Balanceada: {balanced_accuracy_cnb_non_optimized:.4f}; Alpha: {model_cnb.alpha:.3f}\n    \
          Unidade de medida no dataset: {unidade_medida_presente}', fontsize=fonte_do_titulo)  # Aumenta o tamanho da fonte do título
plt.xlabel('Classe Predita', fontsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte do rótulo do eixo x
plt.ylabel('Classe Real', fontsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte do rótulo do eixo y
plt.tick_params(axis='x', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo x
plt.tick_params(axis='y', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo y
plt.show()

# RF
"""
print(confusion_matrix(y_test, y_pred_test_cnb_before))
plt.figure(figsize=(12,10))
sns.heatmap(df_cm_non_optimized_rf_log.astype(int), annot=df_cm_non_optimized_rf, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Cores em Escala Logarítmica'}, annot_kws={"size": fonte_dos_numeros}) 
plt.title(f'Dataset: {dataset_escolhido} - Conjunto de Teste - Antes da Otimização de Hiperparâmetros\n \
          Algoritmo: {type(model_rf)}\nVetorizador: {type(vectorizer)}; N-gramas: {vectorizer.ngram_range}\n \
          Acurácia Balanceada: {balanced_accuracy_rf_non_optimized:.4f}; Alpha: {model_rf.alpha:.3f}\n    \
          Unidade de medida no dataset: {unidade_medida_presente}', fontsize=fonte_do_titulo)  # Aumenta o tamanho da fonte do título
plt.xlabel('Classe Predita', fontsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte do rótulo do eixo x
plt.ylabel('Classe Real', fontsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte do rótulo do eixo y
plt.tick_params(axis='x', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo x
plt.tick_params(axis='y', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo y
plt.show()
"""

end_time_counter(bloco)

# %% - GRIDSEARCH - AJUSTE DE HIPERPARÂMETROS
"""
AJUSTE DOS HIPERPARÂMETROS PELO MÉTODO GRIDSEARCH
"""

bloco = 'AJUSTE DOS HIPERPARÂMETROS'
start_time_counter(bloco)

# Biblioteca para buscar melhores parâmetros: GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, recall_score  # Prevendo implementações futuras

# Definição das faixas de parâmetros para otimização
param_grid_mnb = {
    'alpha': np.linspace(0.01, 1.0, 20),   # Gerar escala linear entre 0.01 a 1.0 com 20 passos
    'fit_prior': [True, False]             # No MNB, para dados desbalanceados, o ideal é False (documentação do Sklearn)
    # 'class_prior': ...   # Por padrão, o modelo calculará as probabilidades a priori das classes com base nos dados de treinamento
    }

param_grid_cnb = {
    'alpha': np.linspace(0.01, 1.0, 20)   # Gerar escala linear entre 0.01 a 1.0 com 20 passos
    }
"""
param_grid_rf = {
    'n_estimators': [100, 200, 300]
    }
"""
# O score ideal para o caso é o que leva em consideração o bom desempenho em TODAS as classes,
# e não somente na majoritária: balanced_accuracy.
meu_scoring = 'balanced_accuracy'  # Opções: 'recall', 'accuracy', 'balanced_accuracy'...

# Instanciar o GS com os classificadores e seus hiperparâmetros (#n_jobs = -1 deu problemas)
# cv=(inteiro) faz com que seja utilizado o stratified k-fold, bom para datasets desbalanceados.
grid_search_mnb = GridSearchCV(model_mnb, param_grid_mnb, scoring=meu_scoring, n_jobs=1, verbose=3, cv=5)   
grid_search_cnb = GridSearchCV(model_cnb, param_grid_cnb, scoring=meu_scoring, n_jobs=1, verbose=3, cv=5)   
# grid_search_rf = GridSearchCV(model_rf, param_grid_rf, scoring=meu_scoring, n_jobs=1, verbose=3, cv=5)  # Limitado devido ao tempo de processsamento elevado para o RF

# Executar o fit() com os dados de treinamento. Processo demorado.
# MNB
print('Otimizando modelo MNB...')
fit_time = time.time()
grid_search_mnb.fit(X_train, y_train)
time_consumed = time.time() - fit_time
optimization_time_mnb = time_consumed
print(f'Otimização modelo MNB finalizada. Tempo consumido (segundos): {time_consumed:.2f}')
# CNB
print('Otimizando modelo CNB...')
fit_time = time.time()
grid_search_cnb.fit(X_train, y_train)
time_consumed = time.time() - fit_time
optimization_time_cnb = time_consumed
print(f'Otimização modelo CNB finalizada. Tempo consumido (segundos): {time_consumed:.2f}')
"""
# RF
print('Otimizando modelo RF...')
fit_time = time.time()
grid_search_rf.fit(X_train, y_train)
time_consumed = time.time() - fit_time
optimization_time_rf = time_consumed
print(f'Otimização modelo RF finalizada. Tempo consumido (segundos): {time_consumed:.2f}')
"""
# Salvando os objetos gridsearchcv dos três modelos; já contém os hiperparâmetros
grid_searches = {
    'GS Multinomial NB': grid_search_mnb,
    'GS Complement NB': grid_search_cnb     #,
    # 'GS Random Forest': grid_search_rf
}

# Salvando só os melhores parâmetros dos três modelos
best_params_gs = {
    'Best Param Multinomial NB': grid_search_mnb.best_params_,
    'Best Param Complement NB': grid_search_cnb.best_params_ #,
    # 'Best Param Random Forest': grid_search_rf.best_params_
}

# Imprimir melhor score e melhores parâmetros
print("\nMelhor score na validação cruzada (métrica '" + meu_scoring + "'), para os dados de TREINO:")
print(f"Multinomial NB: {grid_search_mnb.best_score_:.4f}")
print(f"Complement NB: {grid_search_cnb.best_score_:.4f}")
# print(f"Rndom Forest: {grid_search_rf.best_score_:.4f}")
print("\nMelhores parâmetros: ")
print("Multinomial NB: ", grid_search_mnb.best_params_ )
print("Complement NB: ", grid_search_cnb.best_params_ )
# print("Random Forest: ", grid_search_rf.best_params_ )
		
end_time_counter(bloco)

# %% - GRIDSEARCH - SALVANDO OS RESULTADOS
"""
SALVANDO O MODELO E SEUS MELHORES ESTIMADORES/PARÂMETROS
"""

# Usando o pickle, dump() e load(). 
# Ao invés de salvar os melhores parâmetros, salvarei o gridsearch inteiro.
# Os melhores parâmetros são parte do atributo best_estimator_

# Serialização
with open('modelos_utilizados.pkl', 'wb') as file:
    pickle.dump(modelos, file)
print("Modelos salvos com sucesso!")

with open('gridsearchcvs_com_melhores_parametros.pkl', 'wb') as file: # Salvarei o gridsearchcv inteiro, ele já inclui os melhores parâmetros.
    pickle.dump(grid_searches, file)
print("Objetos GridSearchCV salvos com sucesso!")

# Agora daria para apagar a variável "model" e gs "Variable Explorer", recuperando-as com o código abaixo:

# Desserialização
with open('modelos_utilizados.pkl', 'rb') as file:
    modelos = pickle.load(file)
print("Modelos recuperados com sucesso!")

with open('gridsearchcvs_com_melhores_parametros.pkl', 'rb') as file:
    grid_searches = pickle.load(file)
print("Objetos GridSearchCV recuperados com sucesso!")

# %% - MELHORES CLASSIFICADORES
"""
TREINAMENTO DSO MODELOS COM OS MELHORES PARÂMETROS OBTIDOS PELO GRIDSEARCH
"""

bloco = 'treinamento com os melhores estimadores obtidos no gridsearchcv'
start_time_counter(bloco)

# Obter os melhores estimadores
best_model_mnb = grid_search_mnb.best_estimator_
best_model_cnb = grid_search_cnb.best_estimator_
# best_model_rf = grid_search_rf.best_estimator_

print("Melhores estimadores obtidos, com seus hiperparâmetros:")
print("Multinomial NB: ", best_model_mnb)
print("Complement NB: ", best_model_cnb)
# print("Random Forest: ", best_model_rf)

# Necessário um fit() em cada um dos modelos antes de fazer predições...
best_model_mnb.fit(X_train, y_train)
best_model_cnb.fit(X_train, y_train)
# best_model_rf.fit(X_train, y_train)
print("Treinamento realizado com sucesso!")

end_time_counter(bloco)

# %% - CLASSIFICADORES - PREDIÇÃO FINAL APÓS A OTIMIZAÇÃO DOS HIPERPARÂMETROS E RELATÓRIOS DE DESEMPENHO, MATRIZES DE CONFUSÃO
"""
AVALIAÇÃO (MÉTRICAS) DO MODELO DE PREDIÇÃO ***JÁ OTIMIZADO***
    - Esta avaliação é feita com base em dataset de TESTE.
"""

bloco = 'avaliação do modelo já otimizado'
start_time_counter(bloco)

# Predições após a otimização dos hiperparâmetros - dataset de TREINO
y_pred_train_mnb_after = best_model_mnb.predict(X_train)
y_pred_train_cnb_after = best_model_cnb.predict(X_train)
# y_pred_train_rf_after = best_model_rf.predict(X_train)

# Predições após a otimização dos  - dataset de TESTE
y_pred_test_mnb_after = best_model_mnb.predict(X_test)
y_pred_test_cnb_after = best_model_cnb.predict(X_test)
# y_pred_test_rf_after = best_model_rf.predict(X_test)

print("Previsões com os datasets de treino e teste realizadas com sucesso!")

# Matrizes de confusão do dataset de TREINO
cm_optimized_train_mnb = confusion_matrix(y_train, y_pred_train_mnb_after)
cm_optimized_train_cnb = confusion_matrix(y_train, y_pred_train_cnb_after)
# cm_optimized_train_rf = confusion_matrix(y_train, y_pred_train_rf_after)
# Matrizes de confusão do dataset de TESTE
cm_optimized_test_mnb = confusion_matrix(y_test, y_pred_test_mnb_after)
cm_optimized_test_cnb = confusion_matrix(y_test, y_pred_test_cnb_after)
# cm_optimized_test_rf = confusion_matrix(y_test, y_pred_test_rf_after)

print("Matrizes de confusão dos datasets de treino e teste criadas com sucesso!")

# Aplicar escala logarítmica às matrizes de confusão do dataset de TREINO, 
# pois há uma categoria muito dominante, que prejudica a visualização em heatmap
cm_optimized_train_mnb_log = np.log1p(cm_optimized_train_mnb)
cm_optimized_train_cnb_log = np.log1p(cm_optimized_train_cnb)
# cm_optimized_train_rf_log = np.log1p(cm_optimized_train_rf)

# Aplicar escala logarítmica às matrizes de confusão do dataset de TESTE, 
# pois há uma categoria muito dominante, que prejudica a visualização em heatmap
cm_optimized_test_mnb_log = np.log1p(cm_optimized_test_mnb)
cm_optimized_test_cnb_log = np.log1p(cm_optimized_test_cnb)
# cm_optimized_test_rf_log = np.log1p(cm_optimized_test_rf)

# Transformar em dataframe para usar no heatmap. TREINO. ANNOTATIONS (VALORES NUMÉRICOS).
df_cm_optimized_train_mnb = pd.DataFrame(
    cm_optimized_train_mnb,
    index=encoder.classes_,
    columns=encoder.classes_
)
df_cm_optimized_train_cnb = pd.DataFrame(
    cm_optimized_train_cnb,
    index=encoder.classes_,
    columns=encoder.classes_
)
"""
df_cm_optimized_train_rf = pd.DataFrame(
    cm_optimized_train_rf,
    index=encoder.classes_,
    columns=encoder.classes_
)
"""
# Transformar em dataframe para usar no heatmap. TREINO. CORES.
df_cm_optimized_train_mnb_log = pd.DataFrame(
    cm_optimized_train_mnb_log,
    index=encoder.classes_,
    columns=encoder.classes_
    )
df_cm_optimized_train_cnb_log = pd.DataFrame(
    cm_optimized_train_cnb_log,
    index=encoder.classes_,
    columns=encoder.classes_
    )
"""
df_cm_optimized_train_rf_log = pd.DataFrame(
    cm_optimized_train_rf_log,
    index=encoder.classes_,
    columns=encoder.classes_
    )
"""
# Transformar em dataframe para usar no heatmap. TESTE. ANNOTATIONS (VALORES NUMÉRICOS).
df_cm_optimized_test_mnb = pd.DataFrame(
    cm_optimized_test_mnb,
    index=encoder.classes_,
    columns=encoder.classes_
)
df_cm_optimized_test_cnb = pd.DataFrame(
    cm_optimized_test_cnb,
    index=encoder.classes_,
    columns=encoder.classes_
)
"""
df_cm_optimized_test_rf = pd.DataFrame(
    cm_optimized_test_rf,
    index=encoder.classes_,
    columns=encoder.classes_
)
"""
# Transformar em dataframe para usar no heatmap. TESTE. CORES.
df_cm_optimized_test_mnb_log = pd.DataFrame(
    cm_optimized_test_mnb_log,
    index=encoder.classes_,
    columns=encoder.classes_
    )
df_cm_optimized_test_cnb_log = pd.DataFrame(
    cm_optimized_test_cnb_log,
    index=encoder.classes_,
    columns=encoder.classes_
    )
"""
df_cm_optimized_test_rf_log = pd.DataFrame(
    cm_optimized_test_rf_log,
    index=encoder.classes_,
    columns=encoder.classes_
    )
"""

# Acurácias balanceadas, para o título do plot:

balanced_accuracy_train_mnb_optimized = balanced_accuracy_score(y_train, y_pred_train_mnb_after)
balanced_accuracy_train_cnb_optimized = balanced_accuracy_score(y_train, y_pred_train_cnb_after)
# balanced_accuracy_train_rf_optimized = balanced_accuracy_score(y_train, y_pred_train_rf_after)
    
balanced_accuracy_test_mnb_optimized = balanced_accuracy_score(y_test, y_pred_test_mnb_after)
balanced_accuracy_test_cnb_optimized = balanced_accuracy_score(y_test, y_pred_test_cnb_after)
# balanced_accuracy_test_rf_optimized = balanced_accuracy_score(y_test, y_pred_test_rf_after)


# MATRIZES DE CONFUSÃO APÓS A OTIMIZAÇÃO DOS HIPERPARÂMETROS

# MNB - TREINO
print(confusion_matrix(y_train, y_pred_train_mnb_after))
plt.figure(figsize=(12,10))
sns.heatmap(df_cm_optimized_train_mnb_log.astype(int), annot=df_cm_optimized_train_mnb, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Cores em Escala Logarítmica'}, annot_kws={"size": fonte_dos_numeros})
plt.title(f'Dataset: {dataset_escolhido} - Conjunto de Treino - Após Otimização de Hiperparâmetros\n \
          Algoritmo: {type(model_mnb)}\nVetorizador: {type(vectorizer)}; N-gramas: {vectorizer.ngram_range}\n   \
          Acurácia Balanceada: {balanced_accuracy_train_mnb_optimized:.4f}; Alpha: {best_model_mnb.alpha:.3f}; Fit_prior: {best_model_mnb.fit_prior}\n \
          Unidade de medida no dataset: {unidade_medida_presente}', fontsize=fonte_do_titulo) 
plt.xlabel('Classe Predita', fontsize=fonte_do_xylabel)
plt.ylabel('Classe Real', fontsize=fonte_do_xylabel)
plt.tick_params(axis='x', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo x
plt.tick_params(axis='y', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo y
plt.show()

# CNB - TREINO
print(confusion_matrix(y_train, y_pred_train_cnb_after))
plt.figure(figsize=(12,10))
sns.heatmap(df_cm_optimized_train_cnb_log.astype(int), annot=df_cm_optimized_train_cnb, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Cores em Escala Logarítmica'}, annot_kws={"size": fonte_dos_numeros})
plt.title(f'Dataset: {dataset_escolhido} - Conjunto de Treino - Após Otimização de Hiperparâmetros\n \
          Algoritmo: {type(model_cnb)}\nVetorizador: {type(vectorizer)}; N-gramas: {vectorizer.ngram_range}\n   \
          Acurácia Balanceada: {balanced_accuracy_train_cnb_optimized:.4f}; Alpha: {best_model_cnb.alpha:.3f}\n \
          Unidade de medida no dataset: {unidade_medida_presente}', fontsize=fonte_do_titulo)          
plt.xlabel('Classe Predita', fontsize=fonte_do_xylabel)
plt.ylabel('Classe Real', fontsize=fonte_do_xylabel)
plt.tick_params(axis='x', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo x
plt.tick_params(axis='y', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo y
plt.show()

"""
# RF - TREINO
print(confusion_matrix(y_train, y_pred_train_rf_after))
plt.figure(figsize=(12,10))
sns.heatmap(df_cm_optimized_train_rf_log.astype(int), annot=df_cm_optimized_train_rf, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Cores em Escala Logarítmica'}, annot_kws={"size": fonte_dos_numeros})
plt.title(f'Dataset: {dataset_escolhido} - Conjunto de Treino - Após Otimização de Hiperparâmetros\n \
          Algoritmo: {type(model_rf)}\nVetorizador: {type(vectorizer)}; N-gramas: {vectorizer.ngram_range}\n    \
          Acurácia Balanceada: {balanced_accuracy_train_rf_optimized:.4f}; N_estimators: {best_model_rf.n_estimators}\n \
          Unidade de medida no dataset: {unidade_medida_presente}', fontsize=fonte_do_titulo) 
plt.xlabel('Classe Predita', fontsize=fonte_do_xylabel)
plt.ylabel('Classe Real', fontsize=fonte_do_xylabel)
plt.tick_params(axis='x', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo x
plt.tick_params(axis='y', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo y
plt.show()
"""

# MNB - TESTE
print(confusion_matrix(y_test, y_pred_test_mnb_after))
plt.figure(figsize=(12,10))
sns.heatmap(df_cm_optimized_test_mnb_log.astype(int), annot=df_cm_optimized_test_mnb, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Cores em Escala Logarítmica'}, annot_kws={"size": fonte_dos_numeros})
plt.title(f'Dataset: {dataset_escolhido} - Conjunto de Teste - Após Otimização de Hiperparâmetros\n \
          Algoritmo: {type(model_mnb)}\nVetorizador: {type(vectorizer)}; N-gramas: {vectorizer.ngram_range}\n   \
          Acurácia Balanceada: {balanced_accuracy_test_mnb_optimized:.4f}; Alpha: {best_model_mnb.alpha:.3f}; Fit_prior: {best_model_mnb.fit_prior}\n \
          Unidade de medida no dataset: {unidade_medida_presente}', fontsize=fonte_do_titulo) 
plt.xlabel('Classe Predita', fontsize=fonte_do_xylabel)
plt.ylabel('Classe Real', fontsize=fonte_do_xylabel)
plt.tick_params(axis='x', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo x
plt.tick_params(axis='y', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo y
plt.show()

# CNB - TESTE
print(confusion_matrix(y_test, y_pred_test_cnb_after))
plt.figure(figsize=(12,10))
sns.heatmap(df_cm_optimized_test_cnb_log.astype(int), annot=df_cm_optimized_test_cnb, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Cores em Escala Logarítmica'}, annot_kws={"size": fonte_dos_numeros})
plt.title(f'Dataset: {dataset_escolhido} - Conjunto de Teste - Após Otimização de Hiperparâmetros\n \
          Algoritmo: {type(model_cnb)}\nVetorizador: {type(vectorizer)}; N-gramas: {vectorizer.ngram_range}\n   \
          Acurácia Balanceada: {balanced_accuracy_test_cnb_optimized:.4f}; Alpha: {best_model_cnb.alpha:.3f}; Fit_prior: {best_model_mnb.fit_prior}\n \
          Unidade de medida no dataset: {unidade_medida_presente}', fontsize=fonte_do_titulo)          
plt.xlabel('Classe Predita', fontsize=fonte_do_xylabel)
plt.ylabel('Classe Real', fontsize=fonte_do_xylabel)
plt.tick_params(axis='x', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo x
plt.tick_params(axis='y', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo y
plt.show()

"""
# RF - TESTE
print(confusion_matrix(y_test, y_pred_test_rf_after))
plt.figure(figsize=(12,10))
sns.heatmap(df_cm_optimized_test_rf_log.astype(int), annot=df_cm_optimized_test_rf, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Cores em Escala Logarítmica'}, annot_kws={"size": fonte_dos_numeros})
plt.title(f'Dataset: {dataset_escolhido} - Conjunto de Teste - Após Otimização de Hiperparâmetros\n \
          Algoritmo: {type(model_rf)}\nVetorizador: {type(vectorizer)}; N-gramas: {vectorizer.ngram_range}\n    \
          Acurácia Balanceada: {balanced_accuracy_test_rf_optimized:.4f}; N_estimators: {best_model_rf.n_estimators}\n \
          Unidade de medida no dataset: {unidade_medida_presente}', fontsize=fonte_do_titulo) 
plt.xlabel('Classe Predita', fontsize=fonte_do_xylabel)
plt.ylabel('Classe Real', fontsize=fonte_do_xylabel)
plt.tick_params(axis='x', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo x
plt.tick_params(axis='y', labelsize=fonte_do_xylabel)  # Aumenta o tamanho da fonte dos rótulos das classes no eixo y
plt.show()
"""

# RELATÓRIOS DE CLASSIFICAÇÃO APÓS A OTIMIZAÇÃO DOS HIPERPARÂMETROS

# MNB - TREINO
print("Relatório de classificação para MNB após a otimização - TREINO:")
print(pd.DataFrame(classification_report(
        y_train,
        y_pred_train_mnb_after,
        target_names=encoder.classes_,  # Isso aqui deu o bicho para descobrir
        digits=3,
        output_dict=True,
        zero_division=np.nan            # Para não ficar dando warnings quando há zero predições
        )).transpose()
    )
print(f'Acurácia balanceada MNB - TREINO: {balanced_accuracy_train_mnb_optimized}')

# CNB - TREINO
print("Relatório de classificação para CNB após a otimização - TREINO:")
print(pd.DataFrame(classification_report(
        y_train,
        y_pred_train_cnb_after,
        target_names=encoder.classes_,  # Isso aqui deu o bicho para descobrir
        digits=3,
        output_dict=True,
        zero_division=np.nan            # Para não ficar dando warnings quando há zero predições
        )).transpose()
    )
print(f'Acurácia balanceada CNB - TREINO: {balanced_accuracy_train_cnb_optimized}')
"""
# RF - TREINO
print("Relatório de classificação para RF após a otimização - TREINO:")
print(pd.DataFrame(classification_report(
        y_train,
        y_pred_train_rf_after,
        target_names=encoder.classes_,  # Isso aqui deu o bicho para descobrir
        digits=3,
        output_dict=True,
        zero_division=np.nan            # Para não ficar dando warnings quando há zero predições
        )).transpose()
    )
print(f'Acurácia balanceada RF - TREINO: {balanced_accuracy_train_rf_optimized}')
"""

# MNB - TESTE
print("Relatório de classificação para MNB após a otimização - TESTE:")
print(pd.DataFrame(classification_report(
        y_test,
        y_pred_test_mnb_after,
        target_names=encoder.classes_,  # Isso aqui deu o bicho para descobrir
        digits=3,
        output_dict=True,
        zero_division=np.nan            # Para não ficar dando warnings quando há zero predições
        )).transpose()
    )
print(f'Acurácia balanceada MNB - TESTE: {balanced_accuracy_test_mnb_optimized}')

# CNB - TESTE
print("Relatório de classificação para CNB após a otimização - TESTE:")
print(pd.DataFrame(classification_report(
        y_test,
        y_pred_test_cnb_after,
        target_names=encoder.classes_,  # Isso aqui deu o bicho para descobrir
        digits=3,
        output_dict=True,
        zero_division=np.nan            # Para não ficar dando warnings quando há zero predições
        )).transpose()
    )
print(f'Acurácia balanceada CNB - TESTE: {balanced_accuracy_test_cnb_optimized}')

"""
# RF - TESTE
print("Relatório de classificação para RF após a otimização - TESTE:")
print(pd.DataFrame(classification_report(
        y_test,
        y_pred_test_rf_after,
        target_names=encoder.classes_,  # Isso aqui deu o bicho para descobrir
        digits=3,
        output_dict=True,
        zero_division=np.nan            # Para não ficar dando warnings quando há zero predições
        )).transpose()
    )
print(f'Acurácia balanceada RF - TESTE: {balanced_accuracy_test_rf_optimized}')
"""

end_time_counter(bloco)

# %% - RESUMO FINAL
"""
RESUMO FINAL
"""
bloco = 'RESUMO FINAL'
start_time_counter(bloco)

print('\nRESUMO FINAL:')
print('Vetorizador utilizado: ')
print(type(vectorizer))
print(f"ngram_range: {vectorizer.ngram_range}")

# Construção manual do dataframe de resultados
classificadores = ['Multinomial NB', 'Complement NB']
# classificadores = ['Multinomial NB', 'Complement NB', 'Random Forest']

num_candidates_gs_mnb = len(grid_search_mnb.cv_results_['params'])
num_candidates_gs_cnb = len(grid_search_cnb.cv_results_['params'])
# num_candidates_gs_rf = len(grid_search_rf.cv_results_['params'])

# candidatos_testados_otimizacao = [num_candidates_gs_mnb, num_candidates_gs_cnb, num_candidates_gs_rf]
candidatos_testados_otimizacao = [num_candidates_gs_mnb, num_candidates_gs_cnb]

# splits_cv = [grid_search_mnb.n_splits_, grid_search_cnb.n_splits_, grid_search_rf.n_splits_]
splits_cv = [grid_search_mnb.n_splits_, grid_search_cnb.n_splits_]

# tempo_para_otimizacao = [optimization_time_mnb, optimization_time_cnb, optimization_time_rf]
tempo_para_otimizacao = [optimization_time_mnb, optimization_time_cnb]

# parametros_antes_otimizacao = [model_mnb.alpha, model_cnb.alpha, model_rf.n_estimators]
# parametros_depois_otimizacao = [best_model_mnb.alpha, best_model_cnb.alpha, best_model_rf.n_estimators]
parametros_antes_otimizacao = [model_mnb.alpha, model_cnb.alpha]
parametros_depois_otimizacao = [best_model_mnb.alpha, best_model_cnb.alpha]

fit_prior_antes_otimizacao = [model_mnb.fit_prior , model_cnb.fit_prior]
fit_prior_depois_otimizacao = [best_model_mnb.fit_prior, best_model_cnb.fit_prior]

# balanced_accuracy_antes_otimizacao_treino = [balanced_accuracy_train_mnb_non_optimized, balanced_accuracy_train_cnb_non_optimized, balanced_accuracy_train_rf_non_optimized]
# balanced_accuracy_antes_otimizacao = [balanced_accuracy_mnb_non_optimized, balanced_accuracy_cnb_non_optimized, balanced_accuracy_rf_non_optimized]
balanced_accuracy_antes_otimizacao_treino = [balanced_accuracy_train_mnb_non_optimized, balanced_accuracy_train_cnb_non_optimized]
balanced_accuracy_antes_otimizacao = [balanced_accuracy_mnb_non_optimized, balanced_accuracy_cnb_non_optimized]

# balanced_accuracy_depois_otimizacao_treino = [balanced_accuracy_train_mnb_optimized, balanced_accuracy_train_cnb_optimized, balanced_accuracy_train_rf_optimized]
# balanced_accuracy_depois_otimizacao_teste = [balanced_accuracy_test_mnb_optimized, balanced_accuracy_test_cnb_optimized, balanced_accuracy_test_rf_optimized]
balanced_accuracy_depois_otimizacao_treino = [balanced_accuracy_train_mnb_optimized, balanced_accuracy_train_cnb_optimized]
balanced_accuracy_depois_otimizacao_teste = [balanced_accuracy_test_mnb_optimized, balanced_accuracy_test_cnb_optimized]


resultados_after = {
    'Classificador': classificadores,
    'Candidatos Testados para Otimização': candidatos_testados_otimizacao,
    'Splits na Validação Cruzada': splits_cv,
    'Tempo para Otimização': tempo_para_otimizacao,
    'Parâmetro antes': parametros_antes_otimizacao,
    'Parâmetro depois': parametros_depois_otimizacao,
    'Fit_prior antes': fit_prior_antes_otimizacao,
    'Fit_prior depois': fit_prior_depois_otimizacao,    
    'Acurácia Balanceada (treino) Antes da Otimização': balanced_accuracy_antes_otimizacao_treino,
    'Acurácia Balanceada (teste) Antes da Otimização': balanced_accuracy_antes_otimizacao,
    'Acurácia Balanceada (treino) Depois da Otimização': balanced_accuracy_depois_otimizacao_treino,
    'Acurácia Balanceada (teste) Depois da Otimização': balanced_accuracy_depois_otimizacao_teste
    }

df_resultados_after = pd.DataFrame(resultados_after)
print(df_resultados_after)
nome_arquivo = 'resultados_finais - ' + dataset_escolhido + '.xlsx'
df_resultados_after.to_excel(nome_arquivo, index=False)

end_time_counter(bloco)
# %% - SALVANDO PREDIÇÕES ERRÔNEAS EM ARQUIVO
"""
CRIAÇÃO DE ARQUIVO COM OS TEXTOS CLASSIFICADOS INCORRETAMENTE, PARA FINS DE MELHORIAS NO TRATAMENTO DAS STRINGS
"""

bloco = 'APRESENTAÇÃO DAS CLASSIFICAÇÕES INCORRETAS'
start_time_counter(bloco)

# Reescrevendo o dicionário de modelos, sem o RF
modelos = {
    'Classificador_MNB': model_mnb,
    'Classificador_CNB': model_cnb
}

# Para cada um dos modelos, salva as classificações incorretas
print('\nRealizando predições para o dataset INTEIRO (treino+teste)...')
for nome_modelo, modelo in modelos.items():
    print(f'\nModelo: {nome_modelo}')
    print(type(modelo))
    y_pred = modelo.predict(X)

    # Calcular a quantidade de sentenças totais e de sentenças classificadas incorretamente
    total_processed_lines = len(processed_lines)
    contagem_de_classificacoes_erradas = sum(y[i] != y_pred[i] for i in range(total_processed_lines))
    
    print(f"Total de textos: {total_processed_lines}")
    print(f"Textos classificados incorretamente: {contagem_de_classificacoes_erradas}")
    
    # Obter os textos tokenizados que foram classificados incorretamente
    print("Obtendo a lista de sentenças classificadas erroneamente...")
    sentencas_classificadas_incorretamente = []
    for i in range(len(processed_lines)):
        if y[i] != y_pred[i]:
            # tokenized_text = vectorizer.inverse_transform(X[i])
            tokenized_text = vectorizer.inverse_transform(X[i].toarray())
            sentencas_classificadas_incorretamente.append((processed_lines[i], tokenized_text, y[i], y_pred[i]))
            if i % 100 == 0:
                print(f"Verificando sentença original número {i}...")
    
    # Criar um DataFrame com os resultados
    print("Criando um dataframe da lista...")
    df_sentencas_classificadas_incorretamente = pd.DataFrame(sentencas_classificadas_incorretamente, columns=['Sentença Original', 'Sentença Tokenizada', 'Valor Real', 'Valor Predito'])
    
    # Salvar o DataFrame em um arquivo Excel
    print("Salvando em Excel...")
    nome_arquivo = 'sentencas_classificadas_incorretamente - dataset ' + dataset_escolhido + ' - ' + nome_modelo + '.xlsx'
    df_sentencas_classificadas_incorretamente.to_excel(nome_arquivo , sheet_name=nome_modelo, index=False)
    
print("As sentenças classificadas incorretamente foram salvas no arquivo 'sentencas_classificadas_incorretamente (dataset) (modelo).xlsx'.")

end_time_counter(bloco)