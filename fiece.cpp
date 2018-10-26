#include "fiece.hpp"
#include "populacao.hpp"

#define DEBUG_MAIN 1

using namespace std;
using namespace arma;

/**
 * Função para criar um grafo a partir de um conjunto de restrições
 *
 * @param restricoes mat
 * @param num_objs int
 **/
grafo* criarGrafo(mat restricoes, int num_objs) {
    assert(restricoes.n_rows > 0);
    assert(restricoes.n_cols == 3);
	int num_vertices = num_objs;
	int num_arestas = 0;
    uword k;
    for(k = 0;k < restricoes.n_rows;++k){
        if(restricoes(k, 2) == 1) ++num_arestas;
    }


    grafo* g = (grafo*) malloc( sizeof(grafo) );
    g->V = num_vertices;
    g->E = num_arestas;
    g->arestas = (aresta*) malloc(g->E * sizeof(aresta) );

    int i_aresta = 0;
	for(k = 0; k < restricoes.n_rows; k++) {
        if(restricoes(k,2) != 1) continue;
        int p, q;
        p = restricoes(k, 0);
        q = restricoes(k, 1);
        g->arestas[i_aresta].origem = p;
        g->arestas[i_aresta++].destino = q;
    }

    return g;
}

int Find_Set(subset* subconjuntos, int i) {
    if (subconjuntos[i].pai != i)
        subconjuntos[i].pai = Find_Set(subconjuntos, subconjuntos[i].pai);
    return subconjuntos[i].pai;
}

void Union(subset* subconjuntos, int x, int y) {
    int rx = Find_Set(subconjuntos, x);
    int ry = Find_Set(subconjuntos, y);

    if (subconjuntos[rx].rank < subconjuntos[ry].rank)
        subconjuntos[rx].pai = ry;
    else if (subconjuntos[rx].rank > subconjuntos[ry].rank)
        subconjuntos[ry].pai = rx;
    else {
        subconjuntos[ry].pai = rx;
        subconjuntos[rx].rank++;
    }
}

vec ComponentesConexas(grafo* g){
	int i, j;
	subset subconjuntos[g->V];
    vec r_vetor(g->V);

	for (i = 0; i<g->V; i++) {
        subconjuntos[i].pai = i;
        subconjuntos[i].rank = 0;
        r_vetor(i) = -1;
    }

    for(j = 0; j<g->E; j++) {
        int rx = Find_Set(subconjuntos, g->arestas[j].origem);
        int ry = Find_Set(subconjuntos, g->arestas[j].destino);

        if (rx != ry)
             Union(subconjuntos, rx, ry);
    }

    // Adiciona todos os representantes de cada numero na posicao correspondente no vetor.
    for(j = 0; j<g->V; j++) {
    	r_vetor(j) = Find_Set(subconjuntos, j);
    }

    return r_vetor;
}



/**
 * Obtém os componentes conexos (chunklets) do grafo das restrições ML
 *
 * Componentes conexos com apenas 1 elemento são desconsiderados.
 * Cada elemento do retorno possui um vector com os índices dos objetos
 * no chunklet
 *
 * @param restricoes mat matriz de restrições |R|x3
 * @param num_objs int número de objetos na base de dados
 * @return vector
 */
vector<vector<uword>>
    geraChunklets(mat restricoes, uword num_objs){
    uword k;
	grafo* g = criarGrafo(restricoes, num_objs);
    vec r_vetor = ComponentesConexas(g);

    vec idx_repr_chunklets = sort(idxRepresChunklets(r_vetor));

    vector<vector<uword>> chunklets(idx_repr_chunklets.n_elem);
    for(k = 0; k< idx_repr_chunklets.n_elem; ++k){
        for(uword i = 0; i < r_vetor.n_elem; ++i){
            if(r_vetor(i) == idx_repr_chunklets(k)){
                chunklets[k].push_back(i);
            }
        }
    }

    free(g->arestas);
    free(g);

    return chunklets;
}

/**
 * Extrai os representante de cada chunklet do vetor obtido de ComponentesConexas
 *
 * Transforma a saída do Union-Find em indices de representantes de chunklets
 * TODO: refatorar
 *
 * @param idx_unions vec índices de representantes obtidos pela função ComponentesConexas
 * @return vec vetor de representantes chunklets (tamanho = |C|)
 */
vec idxRepresChunklets(vec representantes){
    vec vetor_retorno(representantes.n_elem);
	vetor_retorno.fill(-1);
	vec vetor_visitado = vetor_retorno;
    uword i, j, k = 0, l = 0, m;
	bool visitado_prim, visitado_sec;
	for(i = 0; i < representantes.n_elem; i++){
		visitado_prim = false;
		for(j = 0; j < k; j++){
			if(representantes(i) == vetor_visitado(j)){
				visitado_prim = true;
				break;
			}
		} //posso tirar esse for
		if(!visitado_prim){
			vetor_visitado(k) = representantes(i);
			k++;
		}
		else{
			visitado_sec = false;
			for(m = 0; m < l; m++){
				if(representantes(i) == vetor_retorno(m)){
					visitado_sec = true;
				}
			}
			if(!visitado_sec){
				vetor_retorno(l) = representantes(i);
				l++;
			}
		}
	}
	vetor_retorno.shed_rows(l, vetor_retorno.n_elem - 1);
	return vetor_retorno;
}


/**
 * Método principal FIECE-EM
 *
 * @param dados mat matriz de objetos NxM
 * @param restricoes mat matriz de restrições objetos |R|x3
 * @param parametros fiecem_prm parâmetros para a execução
 */
Individuo* fieceem(const mat& dados, const mat & restricoes, fiecem_prm parametros){

	//gerando chunklets a partir das restrições
    vector<vector<uword>> chunklets;
    chunklets = geraChunklets(restricoes, dados.n_rows);

    #ifdef DEBUG_MAIN
    cout << "Inicialização das populações" << endl;
    #endif

    Populacao feasible(parametros.tam_populacao);
    Populacao infeasible(parametros.tam_populacao);
	geraPopulacoes(feasible, infeasible, parametros, chunklets, dados);

    #ifdef DEBUG_MAIN
	cout << "individuos em feasible: " << feasible.getNumIndividuos() << endl;
	cout << "individuos em infeasible: " << infeasible.getNumIndividuos() << endl;
    #endif

    for(uword Sc_2 = 0; Sc_2 < parametros.max_geracoes; Sc_2++){
    	Populacao feasible_pool(parametros.tam_populacao);
	    Populacao infeasible_pool(parametros.tam_populacao);

	    for(uword ind = 0; ind < feasible.getNumIndividuos(); ind++){
			Individuo *individuo = feasible.getIndividuo(ind);

			#ifdef DEBUG_MAIN
			cout << "LAÇO FEASIBLE (" << ind << ")\n-------------------\n" << endl;
			individuo->Imprime();
			#endif

	    	individuo->removeGruposVazios(dados);

			#ifdef DEBUG_MAIN
			cout << "   REMOVEU GRUPOS VAZIOS" << endl;
			individuo->Imprime();
			#endif

	    	individuo->rotularGruposComObjetosForaDeChunklets(chunklets, dados);
            #ifdef DEBUG_MAIN
			cout << "   ROTULOU GRUPOS" << endl;
			individuo->Imprime();
			#endif

			individuo->ExpectationMaximization(dados, parametros.num_em_it);
            #ifdef DEBUG_MAIN
			cout << "   RODOU EM" << endl;
			individuo->Imprime();
			#endif


            Individuo *copia = new Individuo(individuo);
            bool added = false;
			if(individuo->ehFeasible(dados, chunklets)){
				#ifdef DEBUG_MAIN
				cout << "Individuo [" << copia << "]adicionado no feasible_pool" << endl;
				#endif
				added = feasible_pool.adicionarIndividuo(copia);
			}else{
				#ifdef DEBUG_MAIN
				cout << "Individuo [" << copia << "]adicionado no infeasible_pool" << endl;
				#endif
				added = infeasible_pool.adicionarIndividuo(copia);
			}
            if(!added) delete copia;

            Individuo *filho = new Individuo(individuo);

	    	filho->mutacaoFeasible(parametros.max_grupos, dados, chunklets);
			#ifdef DEBUG_MAIN
			cout << "MUTACAO FEASIBLE [" <<
                filho->ehFeasible(dados, chunklets)  << "]" << endl;
			filho->Imprime();
			#endif

			filho->ExpectationMaximization(dados, parametros.num_em_it);
            #ifdef DEBUG_MAIN
			cout << "   RODOU EM" << endl;
			filho->Imprime();
			#endif

			if(filho->ehFeasible(dados, chunklets)){
				#ifdef DEBUG_MAIN
				cout << "Individuo [" << filho << "]adicionado no feasible_pool" << endl;
				#endif
				added = feasible_pool.adicionarIndividuo(filho);
			}else{
				#ifdef DEBUG_MAIN
				cout << "Individuo [" << filho << "]adicionado no infeasible_pool" << endl;
				#endif
				added = infeasible_pool.adicionarIndividuo(filho);
			}
            if(!added) delete filho;
			#ifdef DEBUG_MAIN
	    	cout << "FEASIBLE -- Sc_2 = " << Sc_2 << " ind = " << ind << endl;
			#endif
	    }


		for(uword ind = 0; ind < infeasible.getNumIndividuos(); ind++){
			Individuo* filho = new Individuo(infeasible.getIndividuo(ind));

			#ifdef DEBUG_MAIN
			cout << "LAÇO INFEASIBLE (" << ind << ")\n-------------------\n" << endl;
            filho->Imprime();
			#endif

			filho->mutacaoInfeasible(parametros.max_grupos, dados, chunklets);
            #ifdef DEBUG_MAIN
			cout << "   MUTACAO INFEASIBLE" << endl;
			filho->Imprime();
            #endif
            bool added = false;
			if(filho->ehFeasible(dados, chunklets)){
				#ifdef DEBUG_MAIN
				cout << "Individuo [" << filho << "]adicionado no feasible_pool" << endl;
				#endif
				added = feasible_pool.adicionarIndividuo(filho);
			}else{
				#ifdef DEBUG_MAIN
				cout << "Individuo [" << filho << "]adicionado no infeasible_pool" << endl;
				#endif
				added = infeasible_pool.adicionarIndividuo(filho);
			}
            if(!added) delete filho;
			#ifdef DEBUG_MAIN
			cout << "INFEASIBLE -- Sc_2 = " << Sc_2 << " ind_it = " << ind << endl;
			#endif
		}

		#ifdef DEBUG_MAIN
		cout << "AS SELEÇÕES VÃO COMEÇAR" << endl;
		cout << "MI+LAMBDA" << endl;
		cout << "feas_pool num_ind: " << feasible_pool.getNumIndividuos() << " # feas num_ind: " << feasible.getNumIndividuos() << endl;
		#endif
		feasible.selecaoMiMaisLambda(feasible_pool, dados, chunklets);

		#ifdef DEBUG_MAIN
        feasible.ImprimeVetor_individuos();
		cout << "ROLETA" << endl;
		cout << "infeas_pool num_ind: " << infeasible_pool.getNumIndividuos() << " # infeas num_ind: " << infeasible.getNumIndividuos() << endl;
		#endif
		infeasible.selecaoRoleta(infeasible_pool, dados, chunklets);


#ifdef DEBUG_MAIN
        cout << "##FEASIBLE" << endl;
        feasible.ImprimeVetor_individuos();
        cout << "##FEASIBLE POOL" << endl;
        feasible_pool.ImprimeVetor_individuos();
        cout << "##INFEASIBLE" << endl;
        infeasible.ImprimeVetor_individuos();
        cout << "##INFEASIBLE POOL" << endl;
        infeasible_pool.ImprimeVetor_individuos();
#endif

    }
    Individuo* melhor = feasible.melhorFeasible(dados);
    if(melhor == NULL){
        melhor = infeasible.melhorInfeasible(chunklets, dados);
    }

    return new Individuo(melhor);
}

/**
 * Cria novos indivíduos e os coloca na população feasible ou infeasible
 *
 * @param feasible população feasible
 * @param infeasible população infeasible
 * @param parametros parametros do fieceem
 * @param chunklets vetor com os índices dos objetos em cada chunklet
 * @param dados matriz de dados NxM
 *
 */
void geraPopulacoes(Populacao& feasible, Populacao& infeasible,
        fiecem_prm parametros, vector<vector<uword>> chunklets,
        const mat& dados){

    uword min_grupos = chunklets.size();
    int taxa = (parametros.max_grupos - min_grupos)/parametros.tam_populacao;
    uword num_tentativas = 0;
    uword quant_grupos = min_grupos;
    while(true){
#ifdef DEBUG_MAIN
        cout << "tentativa: " << num_tentativas << endl;
        cout << "# feas: " << feasible.getNumIndividuos() << endl;
        cout << "# infeas: " << infeasible.getNumIndividuos() << endl;
#endif
        Individuo* individuo = new Individuo(quant_grupos, chunklets, dados);
        if(individuo->ehFeasible(dados, chunklets)){
            feasible.adicionarIndividuo(individuo);
            num_tentativas = 0;
        }else{
            ++num_tentativas;
            if(infeasible.getNumIndividuos() < parametros.tam_populacao){
                infeasible.adicionarIndividuo(individuo);
            }else{
                delete individuo;
            }
        }

        quant_grupos += taxa;
        if(quant_grupos > parametros.max_grupos){
            quant_grupos = min_grupos + (quant_grupos % parametros.max_grupos);
        }

        if(num_tentativas == parametros.max_tentativas)
            break;
        if(feasible.getNumIndividuos() == parametros.tam_populacao)
            break;
    }
}

