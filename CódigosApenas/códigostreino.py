import pandas as pd
from pycaret.classification import *
from pycaret.classification import get_config

# 1 carregando csv/dataset
df = pd.read_csv('meu_dataset_phishing.csv')

print("CSV carregado com sucesso!")
print("versão original(Inglês-EUA) do dataset:")
print(df.head())

# Dicionário de tradução das colunas
dicionario_colunas = {
    'having_IP_Address': 'tem_endereco_ip',
    'URL_Length': 'comprimento_url',
    'Shortining_Service': 'servico_encurtamento',
    'having_At_Symbol': 'tem_simbolo_arroba',
    'double_slash_redirecting': 'redirecionamento_barra_dupla',
    'Prefix_Suffix': 'prefixo_sufixo',
    'having_Sub_Domain': 'tem_sub_dominio',
    'SSLfinal_State': 'estado_ssl_final',
    'Domain_registeration_length': 'duracao_registro_dominio',
    'Favicon': 'favicon',
    'port': 'porta_nao_padrao',
    'HTTPS_token': 'token_https_no_dominio',
    'Request_URL': 'url_requisicao_externa',
    'URL_of_Anchor': 'url_das_ancoras',
    'Links_in_tags': 'links_em_tags_meta',
    'SFH': 'sfh',
    'Submitting_to_email': 'envio_formulario_para_email',
    'Abnormal_URL': 'url_anormal',
    'Redirect': 'contagem_redirecionamentos',
    'on_mouseover': 'evento_on_mouseover',
    'RightClick': 'clique_direito_desabilitado',
    'popUpWidnow': 'janela_popup',
    'Iframe': 'usa_iframe',
    'age_of_domain': 'idade_dominio',
    'DNSRecord': 'registro_dns',
    'web_traffic': 'trafego_web',
    'Page_Rank': 'page_rank',
    'Google_Index': 'indexado_google',
    'Links_pointing_to_page': 'links_apontando_pagina',
    'Statistical_report': 'relatorio_estatistico',
    'Result': 'Resultado'
}

# Renomeando as colunas do dataset 
df = df.rename(columns=dicionario_colunas)

print("versão traduzida do dataset:")
print(df.head())

# Verificando e tratando valores nulos
print(f"\nValores nulos por coluna:")
print(df.isnull().sum())

# Verificando valores únicos em cada coluna
print(f"\nVerificando valores únicos:")
for col in df.columns:
    valores_unicos = df[col].unique()
    print(f"{col}: {valores_unicos}")

# Removendo linhas com valores que não sejam -1, 0 ou 1
valores_validos = [-1, 0, 1]
for col in df.columns:
    df = df[df[col].isin(valores_validos)]

print(f"\nShape do dataset após limpeza: {df.shape}")

# Iniciando setup do pycaret
print("\nIniciando setup do pycaret...")

experimento_ia = setup(data=df, target='Resultado', session_id=123)

# Análise tabela primeiro setup:
# 0 == Legítimo
# -1 == Phishing
# Uso de todos os núcleos da CPU

# ETAPA 2: Treinar os modelos de IA 
print("\nIniciando treinamento dos modelos...")

# testando diversos modelos
melhores_modelos = compare_models(n_select=5)
print("Melhores modelos:")
print(melhores_modelos)
print("\n")

print("Resultados dos modelos:")
print(pull())

print("\nAvaliando os modelos...")