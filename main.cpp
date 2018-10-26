#include "fiece.hpp"

void Teste_OP_Criacao_inserindo_parametros_para_inicializar_individuo();
void teste_main(arma::mat);

#define DEBUG_MAIN 1

/*
 * Parametros:
 * - tam_pop: quantidade de individuos na populacao;
 * - num_dados: numero de dados coletados/numero de observacoes;
 * - num_atrib: numero de atributos;
 */
int main(){
	int quant_classes, quant_grupos;
	arma::mat mat_rest_ML;
	arma::vec objetos_representantes_objetos, objetos_representantes_dos_chunklets;
	arma::mat* cubo_chunklets;
	arma::mat dados = {{1.76836523945123,  1.35737804545788},
					   {1.37593936283692,  1.71752839397186},
					   {1.79780220329521,  0.649294241602705},
					   {1.69317243390757,  0.505908000256418},
					   {0.195295869058187, 1.60094739752044},
					   {1.12748632839894,  1.14533575664819},
					   {0.981453948163206, 0.877966889354757},
					   {0.600322516300666, 0.546139010816261},
					   {0.322184539140937, 1.05658796423968},
					   {0.139037354015211, 2.31394215687004},
					   {4.39844760208826,  3.29398621895481},
					   {4.11715193230929,  3.89262756603667},
					   {4.99033979366252,  5.14475281232284},
					   {3.79971268377487,  4.77061603027481},
					   {3.36233646618856,  5.04629807137534},
					   {4.09424400169666,  4.90901636485266},
					   {3.45854546733802,  5.0671649558891},
					   {3.90926158107257,  3.50080340661631},
					   {4.10061089763018,  4.151936747684},
					   {2.65490387320801,  4.78247109245389},
					   {10.698855746751,   4.157390689762},
					   {11.2110187097293,  5.20508544226322},
					   {9.58102885476079,  4.15313027877679},
					   {9.38289422347831,  4.45364163975202},
					   {10.9517052565108,  3.79358931730023},
					   {9.22276595063008,  5.15925655977313},
					   {9.40349874254272,  4.48525461160612},
					   {10.1077418197357,  5.28420452204607},
					   {9.80094033364024,  3.45430168055936},
					   {9.97239488328039,  4.2641297137084}};
	//Teste_OP_Criacao_inserindo_parametros_para_inicializar_individuo();
	//teste_main(dados);

    mat_rest_ML = {{1, 2},
				   {2, 4},
				   {2, 8},
				   {15, 13},
				   {22, 23}};

	//gerando chunklets a partir das restrições
	objetos_representantes_objetos = objetosRepresentantesDeObjetos(mat_rest_ML, dados);
	objetos_representantes_dos_chunklets = indicesDosObjetosRepresentantesDosChunklets(mat_rest_ML, dados);
	cubo_chunklets = new arma::mat[objetos_representantes_dos_chunklets.n_elem];
	cubo_chunklets = objetosEAtributosDeTodosOsChunklets(cubo_chunklets, objetos_representantes_dos_chunklets, mat_rest_ML, dados);

	std::cout << "gerou cubo_chunklets\n";
	quant_classes = objetos_representantes_dos_chunklets.n_elem;
	quant_grupos = quant_classes;
	if(quant_classes > quant_grupos){
		std::cout << "AVISO: Quantidade de classes maior que quantidade de grupos" << std::endl;
		abort();
	}
	//inicializando populacoes
	int tam_pop = 2, quant_max_grupos = 8, num_max_tentativas = 30;
    Populacao* feasible;
    Populacao* infeasible; //cria um ponteiro do tipo Populacao
	feasible = new Populacao(0, quant_classes, quant_grupos, objetos_representantes_dos_chunklets, dados, cubo_chunklets); //verificar se violam restricoes
    infeasible = new Populacao(0, quant_classes, quant_grupos, objetos_representantes_dos_chunklets, dados, cubo_chunklets);
	geraPopulacoes(feasible, infeasible, num_max_tentativas, quant_max_grupos, tam_pop, quant_classes, quant_grupos, objetos_representantes_dos_chunklets, dados, cubo_chunklets);
	std::cout << "quant_individuos depois de geraPopulações em feasible: " << feasible->getNumIndividuos() << std::endl;
	std::cout << "quant_individuos depois de geraPopulações em infeasible: " << infeasible->getNumIndividuos() << std::endl;

    int num_geracoes = 2;
    int num_iteracoes_EM = 3;
    for(int Sc_2 = 0; Sc_2 < num_geracoes; Sc_2++){
    	Populacao* feasible_pool;
	    Populacao* infeasible_pool;
	    feasible_pool = new Populacao(0, quant_classes, quant_grupos, objetos_representantes_dos_chunklets, dados, cubo_chunklets);
	    infeasible_pool = new Populacao(0, quant_classes, quant_grupos, objetos_representantes_dos_chunklets, dados, cubo_chunklets);

	    for(int ind_it = 0; ind_it < feasible->getNumIndividuos(); ind_it++){
	    	//TODO devo criar uma copia do individuo ao invés de alterá-lo, correto?
			#if DEBUG_MAIN
			std::cout << "LAÇO FEASIBLE\n-------------------\n" << std::endl;
			#endif
			Individuo* individuo = feasible->getIndividuo(ind_it);
			#if DEBUG_MAIN
			std::cout << "Quant_grupos: " << individuo->getQuantGrupos() << std::endl;
			#endif
			Individuo* copia_individuo = feasible->getIndividuo(ind_it);
	    	individuo->removeGruposVazios(dados);
			#if DEBUG_MAIN
			std::cout << "REMOVEU GRUPOS VAZIOS\n\nQuant_grupos: " << individuo->getQuantGrupos() << std::endl;
			#endif
	    	individuo->rotularGruposComObjetosForaDeChunklets(cubo_chunklets, dados);
			individuo->ExpectationMaximization(dados, num_iteracoes_EM);
			if(individuo->ehFeasible(dados, cubo_chunklets)){
				#if DEBUG_MAIN
				std::cout << "Individuo adicionado no feasible_pool\n\n" << std::endl;
				#endif
				feasible_pool->adicionarIndividuo(individuo);
			}
			else{
				#if DEBUG_MAIN
				std::cout << "Individuo adicionado no infeasible_pool\n\n" << std::endl;
				#endif
				infeasible_pool->adicionarIndividuo(individuo);
			}

	    	copia_individuo->geraMutacaoFeasible((double)quant_max_grupos, dados, objetos_representantes_objetos, objetos_representantes_dos_chunklets);
			copia_individuo->ExpectationMaximization(dados, num_iteracoes_EM);
			#if DEBUG_MAIN
			std::cout << "Gerou as mutacoes\n\n" << std::endl;
			#endif
			if(individuo->ehFeasible(dados, cubo_chunklets)){
				#if DEBUG_MAIN
				std::cout << "Individuo adicionado no feasible_pool\n\n" << std::endl;
				#endif
				feasible_pool->adicionarIndividuo(copia_individuo);
			}
			else{
				#if DEBUG_MAIN
				std::cout << "Individuo adicionado no infeasible_pool\n\n" << std::endl;
				#endif
				infeasible_pool->adicionarIndividuo(copia_individuo);
			}
			#if DEBUG_MAIN
	    	std::cout << "FEASIBLE -- Sc_2 = " << Sc_2 << " ind_it = " << ind_it << std::endl;
			#endif
	    }

		for(int ind_it = 0; ind_it < infeasible->getNumIndividuos(); ind_it++){
			#if DEBUG_MAIN
			std::cout << "LAÇO INFEASIBLE\n-------------------\n" << std::endl;
			#endif
			Individuo* individuo = infeasible->getIndividuo(ind_it);
			#if DEBUG_MAIN
			std::cout << "Quant_grupos: " << individuo->getQuantGrupos() << std::endl;
			#endif
			individuo->geraMutacaoInfeasible((double)quant_max_grupos, dados, objetos_representantes_objetos, objetos_representantes_dos_chunklets, cubo_chunklets);
			if(individuo->ehFeasible(dados, cubo_chunklets)){
				#if DEBUG_MAIN
				std::cout << "Individuo adicionado no feasible_pool\n\n" << std::endl;
				#endif
				feasible_pool->adicionarIndividuo(individuo);
			}
			else{
				#if DEBUG_MAIN
				std::cout << "Individuo adicionado no infeasible_pool\n\n" << std::endl;
				#endif
				infeasible_pool->adicionarIndividuo(individuo);
			}
			#if DEBUG_MAIN
			std::cout << "INFEASIBLE -- Sc_2 = " << Sc_2 << " ind_it = " << ind_it << std::endl;
			#endif
		}

		#if DEBUG_MAIN
		std::cout << "AS SELEÇÕES VÃO COMEÇAR" << std::endl;
		std::cout << "MI+LAMBDA" << std::endl;
		std::cout << "feas_pool num_ind: " << feasible_pool->getNumIndividuos() << " # feas num_ind: " << feasible->getNumIndividuos() << std::endl;
		#endif
		feasible->selecaoMiMaisLambda(feasible_pool, quant_classes, quant_grupos, objetos_representantes_dos_chunklets, dados, cubo_chunklets);
		#if DEBUG_MAIN
		std::cout << "ROLETA" << std::endl;
		std::cout << "infeas_pool num_ind: " << infeasible_pool->getNumIndividuos() << " # infeas num_ind: " << infeasible->getNumIndividuos() << std::endl;
		#endif
		infeasible->selecaoRoleta(infeasible_pool, objetos_representantes_objetos, quant_classes, quant_grupos, objetos_representantes_dos_chunklets, dados, cubo_chunklets);

	    delete feasible_pool;
    	delete infeasible_pool;
    }

    delete infeasible;
    delete feasible;
    return 0;
}




/*
void Teste_OP_Criacao_inserindo_parametros_para_inicializar_individuo(){
	std::cout << "---------------------------------------------------------------------------------------" << std::endl;
	std::cout << "                TESTE INSERINDO PARAMETROS PARA INICIALIZAR O INDIVIDUO                " << std::endl;
	std::cout << "---------------------------------------------------------------------------------------" << std::endl;
	int num_dados, num_atrib, quant_classes, quant_grupos;
	int num_rest_ML, num_rest_CL;
	arma::mat mat_rest_ML, mat_adj_restricoes;
	arma::vec vec_rest_CL, vetor_chunklets;
	arma::mat dados = {{1.76836523945123,  1.35737804545788},
					   {1.37593936283692,  1.71752839397186},
					   {1.79780220329521,  0.649294241602705},
					   {1.69317243390757,  0.505908000256418},
					   {0.195295869058187, 1.60094739752044},
					   {1.12748632839894,  1.14533575664819},
					   {0.981453948163206, 0.877966889354757},
					   {0.600322516300666, 0.546139010816261},
					   {0.322184539140937, 1.05658796423968},
					   {0.139037354015211, 2.31394215687004},
					   {4.39844760208826,  3.29398621895481},
					   {4.11715193230929,  3.89262756603667},
					   {4.99033979366252,  5.14475281232284},
					   {3.79971268377487,  4.77061603027481},
					   {3.36233646618856,  5.04629807137534},
					   {4.09424400169666,  4.90901636485266},
					   {3.45854546733802,  5.0671649558891},
					   {3.90926158107257,  3.50080340661631},
					   {4.10061089763018,  4.151936747684},
					   {2.65490387320801,  4.78247109245389},
					   {10.698855746751,   4.157390689762},
					   {11.2110187097293,  5.20508544226322},
					   {9.58102885476079,  4.15313027877679},
					   {9.38289422347831,  4.45364163975202},
					   {10.9517052565108,  3.79358931730023},
					   {9.22276595063008,  5.15925655977313},
					   {9.40349874254272,  4.48525461160612},
					   {10.1077418197357,  5.28420452204607},
					   {9.80094033364024,  3.45430168055936},
					   {9.97239488328039,  4.2641297137084}};
	num_dados = dados.n_rows;
	num_atrib = dados.n_cols;

	mat_rest_ML = {{1, 2},
				   {2, 4},
				   {2, 8},
				   {15, 13},
				   {22, 23}};
	vec_rest_CL = {1, 3};

	num_rest_ML = (mat_rest_ML.n_rows);
	num_rest_CL = (vec_rest_CL.n_elem)/2;

	mat_adj_restricoes = restricoesToMatrizAdj(mat_rest_ML, num_dados);
	vetor_chunklets = indicesDosObjetosRepresentantesDosChunklets(mat_rest_ML, dados);

	quant_classes = vetor_chunklets.n_elem;
	quant_grupos = quant_classes;
	if(quant_classes > quant_grupos){
		std::cout << "AVISO: Quantidade de classes maior que quantidade de grupos" << std::endl;
		abort();
	}

	arma::vec pesos = {0.3333, 0.3333, 0.3333};
	arma::vec medias[3] = {{10.0333,    4.4410},{3.8886,     4.4560},{1.0001,     1.1771}};
	arma::mat covariancias[3] = {{{0.4418,  -0.0069},
							   {-0.0069,   0.3400}},
							  {{0.3605,   -0.0958},
							   {-0.0958,   0.4280}},
							  {{0.3873,   -0.1529},
							   {-0.1529,   0.3029}}};
	Individuo* ind = new Individuo(quant_classes, quant_grupos, num_atrib, medias, covariancias, pesos);
	ind->ImprimeElementosIndividuo();

	arma::mat matriz_mutacao(quant_grupos, num_atrib);

	std::cout << "-----------------------------------" << std::endl;
	std::cout << "matriz_mutacao:" << std::endl;
	srand(244);
    int maior, menor, i;
    maior = dados.n_rows - 1;
    menor = 0;
    int aleatorio;
	for(i = 0; i < matriz_mutacao.n_rows; i++){
		aleatorio = rand()%(maior-menor+1) + menor;
		matriz_mutacao.row(i) = dados.row(aleatorio);
	}
	std::cout << matriz_mutacao << std::endl;

	ind->Operador_Criar(dados, matriz_mutacao, mat_rest_ML, vetor_chunklets);
	ind->ImprimeElementosIndividuo();
	delete ind;
}*/

/*void teste_main(arma::mat dados){
	int tam_pop = 5;
	int num_dados, num_atrib, quant_classes, quant_grupos;
	int num_rest_ML;
	arma::mat mat_rest_ML, mat_adj_restricoes;
	arma::vec vetor_chunklets;
	num_dados = dados.n_rows;
	num_atrib = dados.n_cols;

	mat_rest_ML = {{1, 2},
				   {2, 4},
				   {2, 8},
				   {15, 13},
				   {22, 23}};

	num_rest_ML = (mat_rest_ML.n_rows);

	mat_adj_restricoes = restricoesToMatrizAdj(mat_rest_ML, num_dados);
	//imprimirMatrizAdjacenteRestricoes(mat_adj_restricoes);
	vetor_chunklets = indicesDosObjetosRepresentantesDosChunklets(mat_rest_ML, dados);
	arma::mat* cubo_chunklets = objetosEAtributosDeTodosOsChunklets(cubo_chunklets, vetor_chunklets, mat_rest_ML, dados);

	std::cout << "chunklets:\n" << vetor_chunklets << std::endl;
	std::cout << "cubo_chunklets:\n";
	for(int i = 0; i < 3; i++){
    	std::cout << cubo_chunklets[i] << std::endl;
    } //TODO

	quant_classes = vetor_chunklets.n_elem;
	quant_grupos = quant_classes; //MUDAR ISSO DEPOIS
	if(quant_classes > quant_grupos){
		std::cout << "AVISO: Quantidade de classes maior que quantidade de grupos" << std::endl;
		abort();
	}

	Individuo* ind = new Individuo(quant_classes, quant_grupos, vetor_chunklets, dados, cubo_chunklets);
	arma::vec vetor_mutacao = {0, 1}; //tratar caso em que o num do grupo é maior que a quantidade de grupos
	ind->ImprimeElementosIndividuo();

	//Operador_Eliminar(ind, vetor_mutacao);
	//ind->ImprimeElementosIndividuo();
	arma::mat matriz_mutacao(quant_grupos, num_atrib);

	std::cout << "-----------------------------------" << std::endl;
	std::cout << "matriz_mutacao:" << std::endl;
	srand(244); //para gerar números aleatórios reais.
    int maior, menor, i;
    maior = dados.n_rows - 1;
    menor = 0;
    int aleatorio;
	for(i = 0; i < matriz_mutacao.n_rows; i++){
		aleatorio = rand()%(maior-menor+1) + menor;
		matriz_mutacao.row(i) = dados.row(aleatorio);
	}
	std::cout << matriz_mutacao;
	std::cout << "-----------------------------------" << std::endl;

	ind->Operador_Criar(dados, matriz_mutacao, mat_rest_ML, vetor_chunklets);
	ind->ImprimeElementosIndividuo();

	Populacao* feasible;
    Populacao* infeasible; //cria um ponteiro do tipo Populacao
    feasible = new Populacao(tam_pop, quant_classes, quant_grupos, vetor_chunklets, dados, cubo_chunklets);
    infeasible = new Populacao(tam_pop, quant_classes, quant_grupos, vetor_chunklets, dados, cubo_chunklets);

    double funObjInf = ind->funcaoObjetivoInfeasible(vetor_chunklets, cubo_chunklets, dados);
    double funObjFeas = ind->funcaoObjetivoFeasible(dados);
    std::cout << "funObjInf: " << funObjInf << std::endl;
    std::cout << "funObjFeas: " << funObjFeas << std::endl;

    delete feasible;
    delete infeasible;
    delete ind;
}*/

//void teste_qualquer(){
	/*Individuo* ind = new Individuo(quant_classes, quant_grupos, num_atrib, medias, covariancias, pesos);
	std::cout << "individuo criado" << std::endl;
	ind->ImprimeElementosIndividuo();

	std::cout << "quantidade de classes:" << ind->getQuantClasses() << std::endl;
	ind->setQuantClasses(6); //muda a quantidade de classes
    std::cout << "quantidade de classes:" << ind->getQuantClasses() << std::endl;
    std::cout << "quantidade de grupos:" << ind->getQuantGrupos() << std::endl;
	ind->setQuantGrupos(7);
	std::cout << "quantidade de atributos:" << ind->getNumAtrib() << std::endl;
    ind->setNumAtrib(4);
    std::cout << "quantidade de atributos:" << ind->getNumAtrib() << std::endl;
    arma::vec novo_peso = {0.4, 0.2, 0.4};
    std::cout << "peso: \n" << ind->getPesos() << std::endl;
    ind->setPesos(novo_peso);
    std::cout << "peso: \n" << ind->getPesos() << std::endl;
    ind->setPesos(0.3, 1);
    std::cout << "peso: \n" << ind->getPesos() << std::endl;
    std::cout << "peso[1]: " << ind->getPesos(1) << std::endl;
    std::cout << "Mapeamento:\n" << ind->getM() << std::endl;
    ind->criaMapeamentoClassesGrupos(4, 4);
    std::cout << "Mapeamento:\n" << ind->getM() << std::endl;
    ind->criaMapeamentoClassesGrupos(5, 4);
    std::cout << "Mapeamento:\n" << ind->getM() << std::endl;
    ind->criaMapeamentoClassesGrupos(4, 5);
    std::cout << "Mapeamento:\n" << ind->getM() << std::endl;
    ind->remove_M_Linha_Coluna(1, 2);
    std::cout << "Mapeamento:\n" << ind->getM() << std::endl;
    ind->remove_M_Linha_Coluna(2, 3);
    std::cout << "Mapeamento:\n" << ind->getM() << std::endl;
    ind->adiciona_M_Linha_Coluna(2, 3);
    std::cout << "Mapeamento:\n" << ind->getM() << std::endl;
    ind->setElementoM(1, 2, 3);
    std::cout << "Mapeamento:\n" << ind->getM() << std::endl;
    std::cout << "Covariancia: \n" << ((ind->getParametro()).Component(2)).Covariance() << std::endl;
    /*
        void alteraParametros(arma::vec, arma::mat, int); //dado um vetor de
        //medias e uma matriz de covariancia, altera os dados parametros
        //do individuo numa posicao passada como parametro da funcao
        void removeParametros(int); //remove os parametros do individuo
        //numa dada posicao passada como parametro para a funcao
        arma::Row<unsigned int> classificaDados(arma::mat, arma::Row<unsigned int>);
        //dada uma matriz de dados, classifica cada elemento como pertencente
        //a um dado grupo e insere as componentes no vetor passado como
        //parametro da funcao
        gmm::GMM reconstroiGMM(gmm::GMM, int); //reconstroi o gmm passado
        //como parametro a partir do novo numero de grupos, também como parametro
        //da funcao
        double getProbabilidadeGMM(arma::vec, int);
*/
	/*
	arma::mat matriz_mutacao(quant_grupos, num_atrib);

	std::cout << "-----------------------------------" << std::endl;
	std::cout << "matriz_mutacao:" << std::endl;
    int maior, menor, i;
    srand(244);
    maior = dados.n_rows;
    menor = 0;
    int aleatorio;
	for(i = 0; i < matriz_mutacao.n_rows; i++){
		aleatorio = rand()%(maior-menor) + menor;
		matriz_mutacao.row(i) = dados.row(aleatorio);
	}
	std::cout <<  matriz_mutacao << std::endl;
	*/

	//Operador_Criar(dados, ind, matriz_mutacao, mat_rest_ML, vetor_chunklets);
	//ind->ImprimeElementosIndividuo();
//}

