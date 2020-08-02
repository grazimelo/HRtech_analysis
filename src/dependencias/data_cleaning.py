def remover_duplicatas(df):
    #Verificando duplicatas nas linhas
    print('Removendo...')
    df.drop_duplicates(inplace=True)
    #Verificando duplicatas colunas
    df_T = df.T
    print(f'Existem {df_T.duplicated().sum()} colunas duplicadas e {df.duplicated().sum()} linhas duplicadas')
    list_duplicated_columns = df_T[df_T.duplicated(keep=False)].index.tolist()
    df_T.drop_duplicates(inplace = True)
    print('Colunas duplicadas:')
    print(list_duplicated_columns)
    return  df_T.T, list_duplicated_columns

def converter_tipos(df, dicionario_tipo):
    # varrer todas as colunas
    for column in df.columns:
        # converte coluna para tipo indicado no dicionário
        df[column] = df[column].astype(dicionario_tipo[column], errors='ignore')
    return df