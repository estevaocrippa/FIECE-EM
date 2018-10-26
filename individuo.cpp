#include "individuo.hpp"
#include <mlpack/core.hpp>
#include <mlpack/methods/gmm/gmm.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <cmath>

//#define DEBUG_IND 1

using namespace mlpack::gmm;
using namespace mlpack::kmeans;
using namespace mlpack::metric;
using namespace mlpack::distribution;
using namespace mlpack::util;
using namespace arma;
using namespace std;

typedef KMeans<SquaredEuclideanDistance> KMeansType;

Individuo::Individuo()
    : quant_classes(0),
      quant_grupos(0),
      num_atrib(0),
      mapeamento_classes_grupos(0,0),
      parametros(){}


Individuo::Individuo(Individuo* outro)
    : quant_classes(outro->getQuantClasses()),
      quant_grupos(outro->getQuantGrupos()),
   	  num_atrib(outro->num_atrib),
      mapeamento_classes_grupos(quant_classes, quant_grupos),
      parametros(){

	uword k;
    std::vector<GaussianDistribution> vetor;
    for(k = 0; k < quant_grupos; k++){
        vetor.push_back(GaussianDistribution(outro->getParametros().Component(k).Mean(),
                    outro->getParametros().Component(k).Covariance()));
    }
    parametros = GMM(vetor, outro->getPesos());
    setMapeamentoClassesGrupos(outro->getMapeamentoClassesGrupos());
}

Individuo::Individuo(uword quant_grupos,
        std::vector<std::vector<uword>> chunklets, 
        const mat& dados)
    : quant_classes(chunklets.size()),
      quant_grupos(quant_grupos),
   	  num_atrib(dados.n_cols),
      mapeamento_classes_grupos(quant_classes, quant_grupos),
      parametros() {
    assert (quant_grupos < dados.n_rows);
    vec pesos(quant_grupos);
    mat covariancias[quant_grupos];
    vec medias[quant_grupos];

    mat centroides(quant_grupos, dados.n_cols);
    umat map = geraCentroides(centroides, dados, quant_grupos, chunklets);
    setMapeamentoClassesGrupos(map);

    Row<size_t> mapeamento_objetos_grupos(dados.n_rows);
    //KMeans<EuclideanDistance> k(100);
    KMeans<EuclideanDistance> k(5);
    mat centroides_t = centroides.t();
    mat dados_t = dados.t();
    k.Cluster(dados_t, quant_grupos, mapeamento_objetos_grupos,
            centroides_t, false, true);

    for(uword i = 0; i < quant_grupos; i++){
    	mat objetos_no_grupo_i(0, dados.n_cols);
        uword num_objs_mapeados = 0;
        for(uword j = 0; j < dados.n_rows; j++){
            if(mapeamento_objetos_grupos(j) == i){
            	objetos_no_grupo_i.insert_rows(objetos_no_grupo_i.n_rows, dados.row(j));
                num_objs_mapeados++;
            }
        }
        pesos(i) = num_objs_mapeados*1./dados.n_rows;

        medias[i] = centroides_t.col(i);

        //no caso de apenas um objeto a matriz de covariância não é bem-definida
        if(num_objs_mapeados > 1){
            covariancias[i] = arma::cov(objetos_no_grupo_i);
        }else{
            covariancias[i] = mat(dados.n_cols, dados.n_cols, fill::zeros);
        }
        covariancias[i].diag() += 1E-5;
        assert(covariancias[i].n_rows == dados.n_cols);
        assert(covariancias[i].n_cols == covariancias[i].n_rows);
    }

    std::vector<GaussianDistribution> vetor;
    for(uword k = 0; k < quant_grupos; k++){
        vetor.push_back(GaussianDistribution(medias[k], covariancias[k]));
    }
    this->parametros = GMM(vetor, pesos);
}

Individuo::Individuo(int quant_classes, int quant_grupos, int num_atrib, vec* medias, mat* covariancias, vec pesos)
    : quant_classes(quant_classes),
      quant_grupos(quant_grupos),
   	  num_atrib(num_atrib),
      mapeamento_classes_grupos(quant_classes, quant_grupos),
      parametros() {

    criaMapeamentoClassesGrupos();
	int k;
    std::vector<GaussianDistribution> vetor;
    for(k = 0; k < quant_grupos; k++){
        vetor.push_back(GaussianDistribution(medias[k], covariancias[k]));
    }
    this->parametros = GMM(vetor, pesos);
}

Individuo::~Individuo(){
#ifdef DEBUG_IND
    printf("destruindo um individuo [%p]\n", this);
#endif
}

void Individuo::setQuantClasses(uword quant_classes){
    this->quant_classes = quant_classes;
}

uword Individuo::getQuantClasses(){
    return this->quant_classes;
}

void Individuo::setQuantGrupos(uword quant_grupos){
    this->quant_grupos = quant_grupos;
}

uword Individuo::getQuantGrupos(){
    return this->quant_grupos;
}

void Individuo::setNumAtrib(uword num_atrib){
    this->num_atrib = num_atrib;
}

uword Individuo::getNumAtrib(){
    return this->num_atrib;
}

void Individuo::setPesos(vec novo_peso){
    std::vector<GaussianDistribution> vetor;
    for(uword k = 0; k < getQuantGrupos(); k++){
        vetor.push_back(this->parametros.Component(k));
    }
    this->parametros = GMM(vetor, novo_peso);
}

void Individuo::setPesos(double novo_peso, uword i){
    vec peso_atualizado = getPesos();
    assert(i < peso_atualizado.n_rows);
    peso_atualizado(i) = novo_peso;
    std::vector<GaussianDistribution> vetor;
    for(uword k = 0; k < getQuantGrupos(); k++){
        vetor.push_back(this->parametros.Component(k));
    }
    this->parametros = GMM(vetor, peso_atualizado);
}

double Individuo::getPesos(int posicao){
	return this->parametros.Weights()[posicao];
}

vec& Individuo::getPesos(){
    return this->parametros.Weights();
}

void Individuo::setMapeamentoClassesGrupos(umat mapeamento){
    this->mapeamento_classes_grupos = umat(mapeamento);
}

const umat& Individuo::getMapeamentoClassesGrupos(){
    return this->mapeamento_classes_grupos;
}

uword Individuo::getMapeamentoClassesGrupos(int i, int j){
    return this->mapeamento_classes_grupos(i, j);
}

void Individuo::removeLinhaOuColunaDoMapeamentoClassesGrupos(char linha_ou_coluna, uword posicao){
    if(linha_ou_coluna == 'l'){
        if(posicao >= 0 && posicao < this->mapeamento_classes_grupos.n_rows){
            this->mapeamento_classes_grupos.shed_row(posicao);
        }
        else{
            fprintf(stderr, "\nERRO!\nFunção removeLinhaOuColunaDoMapeamentoClassesGrupos\nAcesso a elemento fora da linha\n\n");
            abort();
        }
    }
    else if(linha_ou_coluna == 'c'){
        if(posicao >= 0 && posicao < this->mapeamento_classes_grupos.n_cols){
            this->mapeamento_classes_grupos.shed_col(posicao);
        }
        else{
            fprintf(stderr, "\nERRO!\nFunção removeLinhaOuColunaDoMapeamentoClassesGrupos\nAcesso a elemento fora da coluna\n\n");
            abort();
        }
    }
}

void Individuo::adicionaLinhaOuColunaNoMapeamentoClassesGrupos(char linha_ou_coluna, uword posicao){
    if(linha_ou_coluna == 'l'){
        if(posicao >= 0 && posicao <= this->mapeamento_classes_grupos.n_rows){
            this->mapeamento_classes_grupos.insert_rows(posicao, 1);
        }
        else{
            fprintf(stderr, "\nERRO!\nFunção adicionaLinhaOuColunaNoMapeamentoClassesGrupos\nAcesso a elemento fora da linha\n\n");
            abort();
        }
    }
    else if(linha_ou_coluna == 'c'){
        if(posicao >= 0 && posicao <= this->mapeamento_classes_grupos.n_cols){
            this->mapeamento_classes_grupos.insert_cols(posicao, 1);
        }
        else{
            fprintf(stderr, "\nERRO!\nFunção adicionaLinhaOuColunaNoMapeamentoClassesGrupos\nAcesso a elemento fora da coluna\n\n");
            abort();
        }
    }
}

void Individuo::setElementoM(uword linha, uword coluna, uword elemento){
    if(linha < this->mapeamento_classes_grupos.n_rows && coluna < this->mapeamento_classes_grupos.n_cols){
        this->mapeamento_classes_grupos(linha, coluna) = elemento;
    }
    else{
    	fprintf(stderr, "\nERRO!\nAcesso a elemento fora da matriz\n");
    	abort();
    }
}

const GMM& Individuo::getParametros(){
    return this->parametros;
}

vec* Individuo::getMedias(){
    vec* vetor_retorno = new vec[this->parametros.Gaussians()];
    for(uword i = 0; i < this->parametros.Gaussians(); i++){
        vetor_retorno[i] = this->parametros.Component(i).Mean();
    }
    return vetor_retorno;
}

const vec& Individuo::getMedia(int i){
    return this->parametros.Component(i).Mean();
}

mat* Individuo::getCovariancias(){
    mat* matriz_retorno = new mat[this->parametros.Gaussians()];
    for(uword i = 0; i < this->parametros.Gaussians(); i++){
        matriz_retorno[i] = this->parametros.Component(i).Covariance();
    }
    return matriz_retorno;
}

const mat& Individuo::getCovariancias(int i){
    return this->parametros.Component(i).Covariance();
}

void Individuo::adicionaParametros(vec media, mat covariancia, double peso){
    assert(covariancia.n_rows == media.n_rows);
    assert(covariancia.n_cols == covariancia.n_rows);

    std::vector<GaussianDistribution> vetor;
    vec pesos = zeros<vec>(getQuantGrupos() + 1);
    uword k;
    for(k = 0; k < getQuantGrupos(); k++){
        pesos(k) = getPesos(k);
        vetor.push_back(this->parametros.Component(k));
    }
    vetor.push_back(GaussianDistribution(media, covariancia));
    pesos(k) = peso;
    assert(abs(accu(pesos)-1) < 1e-6);

    this->parametros = GMM(vetor, pesos);
	setQuantGrupos(getQuantGrupos()+1);
}


void Individuo::removeParametros(uword posicao){
	std::vector<GaussianDistribution> vetor;
	vec pesos(getQuantGrupos() - 1);
    double soma = 0;
    for(uword i = 0, j = 0; i < getQuantGrupos(); i++){
        if (i == posicao)
            continue;
        vetor.push_back(this->parametros.Component(i));
        pesos(j) = getPesos(i);
        soma += pesos(j);
        ++j;
    }
    pesos /= soma;
    this->parametros = GMM(vetor, pesos);

    assert(abs(accu(getPesos())-1) < 1e-6);
}

/**
 * Função para gerar centróides iniciais com a matriz de mapeamento
 *
 * Os centróides são retornados por meio do parâmetro centroides enquanto
 * que a matriz de mapeamento é o valor de retorno da função.
 *
 * @param matriz onde os centróides devem ser armazenados
 * @param matriz de dados
 * @param número de grupos
 * @param chunklets
 * @return matriz de mapeamento
 */
umat Individuo::geraCentroides(mat& centroides, const mat& dados, uword n_grupos,
        std::vector<std::vector<uword>> chunklets){
	int idx_aleat, cls_aleat;
    bool usados[dados.n_rows] = {false};
    umat mapeamento_classes_grupos = umat(chunklets.size(), n_grupos, fill::zeros);

    #ifdef DEBUG_IND
    cout << "   Gerando centróides" << endl;
    #endif

    for(uword i = 0; i < n_grupos; i++){
        if(i < chunklets.size()){
            cls_aleat = i;
            idx_aleat = rand()%chunklets[i].size();
            idx_aleat = chunklets[i][idx_aleat];
        }else{
            do{
                idx_aleat = rand()%dados.n_rows;
            }while(usados[idx_aleat]);
            cls_aleat = idxChunkletMaisProximo(chunklets,
                    dados.row(idx_aleat), dados);
        }
        usados[idx_aleat] = true;
        centroides.row(i) = dados.row(idx_aleat);
        mapeamento_classes_grupos(cls_aleat, i) = 1;

        #ifdef DEBUG_IND
        cout << "Grupo " << i << " Classe " << cls_aleat <<
            " Obj " << idx_aleat << endl;
        #endif
    }

    return mapeamento_classes_grupos;

}

void Individuo::criaMapeamentoClassesGrupos(){
    this->mapeamento_classes_grupos.set_size(getQuantClasses(), getQuantGrupos());
    this->mapeamento_classes_grupos.zeros();

    int max_ind_classe, min_ind_classe, aleatorio;
    max_ind_classe = quant_classes - 1;
    min_ind_classe = 0;

    colvec soma_das_linhas;
    for(uword i = 0; i < quant_grupos; i++){
        if(i < quant_classes){
            this->mapeamento_classes_grupos(i, i) = 1;
        }
        else{
            aleatorio = rand() % (max_ind_classe - min_ind_classe + 1) + min_ind_classe;
            this->mapeamento_classes_grupos(aleatorio, i) = 1;
        }
    }
}

mat Individuo::mapeiaObjetosEGrupos_matriz(mat dados, int quant_grupos){
	mat retorno(dados.n_rows, quant_grupos, fill::zeros);
    Row<size_t> labels;
    classificaDados(dados,labels);
	for(uword i = 0; i < labels.n_elem; i++){
		retorno(i, labels(i)) = 1;
	}
	return retorno;
}

void Individuo::classificaDados(const mat& dados, Row<size_t>& labels){
    this->parametros.Classify(dados.t(), labels);
}

double Individuo::getProbabilidadeGMM(vec objeto, int grupo){
    return this->parametros.Probability(objeto, grupo);
}

bool Individuo::objetoEmGrupoCorreto(int idx_chunklet, int label_obj){
    umat matriz_m = getMapeamentoClassesGrupos();
    return matriz_m(idx_chunklet, label_obj) == 1;
}

/**
 * Computa o custo de violacao da restrição do objeto n_esimoOjbeto
 *
 * @param objeto que deve ser considerado para o custo
 * @param cluster que o objeto foi agrupado
 * @param chunklet em que o objeto está
 * @param matriz de mapeamento
 * @return custo da violação
 *
 **/
double Individuo::custoDeViolacaoInfeasible(rowvec n_esimoObjeto, int c_n, int i){
    double valor_retorno;
    umat matriz_M = getMapeamentoClassesGrupos();
    if(objetoEmGrupoCorreto(i, c_n)){
            return 0;
    }

    double maximo = 0;
    for(uword j = 0; j < matriz_M.n_cols; j++){
        double probabilidade = getProbabilidadeGMM(n_esimoObjeto.t(), j);
        if(matriz_M(i, j) == 1 && probabilidade > maximo){
            maximo = probabilidade;
        }
    }
    valor_retorno = 1 - maximo;
    return valor_retorno;
}

bool Individuo::ehFeasible(mat dados, std::vector<std::vector<uword>> chunklets){
    Row<size_t> labels;
    classificaDados(dados,labels);
    for(uword i = 0; i < chunklets.size(); ++i){
        std::vector<uword> objs_ith_chunklet = chunklets[i];
        for(uword j = 0; j < objs_ith_chunklet.size(); j++){
            if(!objetoEmGrupoCorreto(i, (int)labels[objs_ith_chunklet[j]])){
                #ifdef DEBUG_IND
                std::cout << "   [ehF] obj " << objs_ith_chunklet[j] <<
                    " grupo " << labels[objs_ith_chunklet[j]] << endl;
                #endif
                return false;
            }
        }
    }
    return true;
}


double Individuo::funcaoObjetivoInfeasible(std::vector<std::vector<uword>> chunklets, const mat& dados){
    double soma = 0;
    Row<size_t> labels;
    classificaDados(dados,labels);
    for(uword i = 0; i < chunklets.size(); i++){
        for(uword j = 0; j < chunklets[i].size(); j++){
            soma += custoDeViolacaoInfeasible(dados.row(chunklets[i][j]), (int)labels(j), i);
        }
    }
    return soma;
}


/**
 * Computa a responsabilidade do componente ind_grupo sobre o objeto ind_objeto
 *
 * Em outras palavras, computa a probabilidade posteriori de um objeto 
 * para uma determinada componente
 *
 * @param objeto a ser examinado
 * @param indice do grupo a ser avaliado
 * @return responsabilidade da componente sobre o objeto
 **/
double Individuo::responsabilidade(const vec & obj, uword ind_grupo){
    double num = log(getPesos(ind_grupo)) +
        this->parametros.Component(ind_grupo).LogProbability(obj);
    double termos_denom[getQuantGrupos()] = {0};
    double max = -1, soma = 0;

    for(uword j = 0; j < getQuantGrupos(); j++){
        termos_denom[j] = log(getPesos(j)) +
            this->parametros.Component(j).LogProbability(obj);
        if(termos_denom[j] > max) max = termos_denom[j];
    }
    for(uword j = 0; j < getQuantGrupos(); j++){
        soma += exp(termos_denom[j] - max);
    }
    double resp = exp(num - (max + log(soma)));
    assert(!isnan(resp));

    return resp;
}

double Individuo::Likelihood(const mat & dados){
    double retorno = 0, soma = 0, peso_j, normal_j;
    for(uword i = 0; i < dados.n_rows; i++){
        soma = 0;
        for(uword j = 0; j < getQuantGrupos(); j++){
            peso_j = getPesos(j);
            normal_j = this->parametros.Component(j).Probability(dados.row(i).t());
            soma += peso_j * normal_j;
        }
        retorno += log(soma);
    }
    return retorno;
}

double Individuo::funcaoObjetivoFeasible(const mat& dados){
    double retorno, likelihood, n, quant_atributos, quant_grupos;
    n = dados.n_rows;
    quant_atributos = getNumAtrib();
    quant_grupos = getQuantGrupos();
    likelihood = Likelihood(dados);
    retorno = - likelihood + (quant_grupos/2) * (1 + quant_atributos + (quant_atributos * (quant_atributos+1)/2)) * log10(n);

#ifdef DEBUG_IND
    std::cout << "loglikelihood: " << likelihood << std::endl 
        << "mdl: " << retorno;
#endif
    return retorno;
}

void Individuo::removeGruposVazios(mat dados){
	Row<size_t> rotulos;
	classificaDados(dados, rotulos);
    int freq[getQuantGrupos()] = {0};
	for(uword i = 0; i < dados.n_rows; i++){
        freq[rotulos(i)]++;
    }
	for(uword k = 0; k < getQuantGrupos(); k++){
		if(freq[k] == 0 && !unicoGrupoClasse(k)){
			removeParametros(k);
			removeLinhaOuColunaDoMapeamentoClassesGrupos('c', k);
			setQuantGrupos(getQuantGrupos() - 1);
		}
	}
}

/**
 * Retorna o índice do chunklet de um objeto.
 *
 * Caso ele não esteja em nenhum chunklet retorna -1.
 *
 * @param chunklets
 * @param índice do objeto
 * @return índice do chunklet
 **/
int Individuo::chunkletDoObj(std::vector<std::vector<uword>> chunklets, uword objeto){
	for(uword i = 0; i < chunklets.size(); i++){
		for(uword j = 0; j < chunklets[i].size(); j++){
			if(chunklets[i][j] == objeto){
                return i;
			}
		}
	}
	return -1;
}


bool Individuo::objetoEstaNosChunklets(std::vector<std::vector<uword>> chunklets, uword objeto){
    return chunkletDoObj(chunklets, objeto) != -1;
}

/**
 * Retorna o índice do chunklet (classe) mais próxima de um ponto
 *
 * Este ponto pode ser a média de um grupo. O chunklet mais próximo é
 * determinado pelo chunklet com um objeto mais próximo do ponto.
 *
 * @param chunklets
 * @param ponto a ser verificado
 * @param matriz de dados
 * @return índice do chunklet/classe mais próxima
 **/
int Individuo::idxChunkletMaisProximo(std::vector<std::vector<uword>> chunklets, rowvec objeto, const mat &dados){
	int ind_chunklet = -1;
	double menor_distancia = 9999, distancia;
	for(uword i = 0; i < chunklets.size(); i++){
		for(uword j = 0; j < chunklets[i].size(); j++){
			distancia = EuclideanDistance::Evaluate(dados.row(chunklets[i][j]),
                    objeto);
			if(distancia < menor_distancia){
				menor_distancia = distancia;
				ind_chunklet = i;
			}
		}
	}
    assert(ind_chunklet > -1);
	return ind_chunklet;
}

void Individuo::zeraColunaESetElementoM(int linha, int coluna, int elemento){
	this->mapeamento_classes_grupos.col(coluna).zeros();
	setElementoM(linha, coluna, elemento);
}

void Individuo::rotularGruposComObjetosForaDeChunklets(std::vector<std::vector<uword>> chunklets, const mat & dados){
	Row<size_t> rotulos;
	classificaDados(dados, rotulos);
	int ind_classe;
	for(uword i = 0; i < getQuantGrupos(); i++){
		bool grupo_esta_vazio = true;
		for(uword j = 0; j < dados.n_rows && grupo_esta_vazio; j++){
			if(rotulos(j) == i && objetoEstaNosChunklets(chunklets, j)){
					grupo_esta_vazio = false;
                    break;
			}
		}
		if(grupo_esta_vazio){
			ind_classe = idxChunkletMaisProximo(chunklets,getMedia(i).t(),dados);
			zeraColunaESetElementoM(ind_classe, i, 1);
		}
	}
}

double Individuo::ExpectationMaximization(const mat & dados, int num_iteracoes_EM){
    //EMFit<KMeansType, PositiveDefiniteConstraint> em(num_iteracoes_EM, 1e-6);
    EMFit<KMeansType, DiagonalConstraint> em(num_iteracoes_EM);
    double loglikelihood;
    assert(abs(accu(getPesos()) - 1) < 1e-6);
    loglikelihood = this->parametros.Train(dados.t(), 1, true, em);
    //TODO verificar na biblioteca porque isso ocorre
    vec pesos = getPesos()/accu(getPesos());
    setPesos(pesos);

    //assert(abs(accu(getPesos()) - 1) < 1e-6);
#ifdef DEBUG_IND
    std::cout << "likelihood"<< loglikelihood;
#endif
    return loglikelihood;
}

double Individuo::PartialLogLikelihood(uword idx_grupo, const mat& dados){
    double retorno = 0, resp;
    auto cmp = this->parametros.Component(idx_grupo);
    for(uword i = 0; i < dados.n_rows; i++){
        resp = responsabilidade(dados.row(i).t(), idx_grupo);
        retorno += resp * (log(getPesos(idx_grupo)) + cmp.LogProbability(dados.row(i).t()));
    }
    return retorno;
}

/**
 * Seleção de quais grupos devem ser removidos baseado no PLL e roleta
 *
 * @param quantidade mínima de grupos
 * @param matriz de dados
 * @return indíces dos grupos a serem removidos
 *
 **/
vec Individuo::selecaoFeasibleEliminacao(int quant_min_grupos, const mat & dados){
    uword quant_iteracoes = getQuantGrupos() - quant_min_grupos;
    colvec idx_grupos(getQuantGrupos());
    colvec pll_grupos(getQuantGrupos());
    for(uword i = 0; i < getQuantGrupos(); i++){
        idx_grupos(i) = i;
        pll_grupos(i) = PartialLogLikelihood(i, dados);
    }

    for(uword i = 0; i < pll_grupos.n_elem; i++){
        for(uword j = 0; j < pll_grupos.n_elem; j++){
            if(pll_grupos(i) > pll_grupos(j)){
                pll_grupos.swap_rows(i,j);
                idx_grupos.swap_rows(i,j);
            }
        }
    }
    #if DEBUG_IND
    std::cout << "FEASIBLE -- ORDENOU do menor pro maior \n";
    std::cout << "pll_grupos:\n" << pll_grupos << std::endl;
    #endif

    vec probabilidades(getQuantGrupos());
    vec intervalos(getQuantGrupos());
    double soma = getQuantGrupos()*(getQuantGrupos() + 1)/2;

    for(uword i = 0; i < getQuantGrupos(); i++){
        probabilidades[i] = (i+1)/soma;
        intervalos[i] = probabilidades[i];
        if(i > 0)
            intervalos[i] += intervalos[i-1];

    }

    vec retorno(quant_iteracoes, fill::zeros);

    double num_aleatorio;
    for(uword i = 0; i < quant_iteracoes; i++){
        num_aleatorio = (rand() % 101)/100;
        for(uword j = 0; j < intervalos.n_elem; j++){
            if(num_aleatorio < intervalos[j]){
                retorno(i) = idx_grupos(j);
                idx_grupos.shed_row(j);
                pll_grupos.shed_row(j);
                probabilidades.shed_row(j);
                intervalos.shed_row(j);
                double soma_probabilidades = sum(probabilidades);
                for(uword k = 0; k < probabilidades.n_elem; k++){
                    probabilidades(k) /= soma_probabilidades;
                    intervalos[k] = probabilidades[k];
                    if(k > 0){
                        intervalos[k] += intervalos[k-1];
                    }
                }
                break;
            }
        }
    }

    return retorno;
}

mat Individuo::selecaoFeasibleCriacao(int quant_max_grupos, const mat & dados){
    uword quant_iteracoes = quant_max_grupos - getQuantGrupos();
    colvec todos_os_objetos(dados.n_rows);
    colvec entropia_responsabilidades(dados.n_rows);

    for(uword i = 0; i < dados.n_rows; i++){
        todos_os_objetos(i) = i;
        entropia_responsabilidades(i) = 0;
        for(uword j = 0; j < getQuantGrupos(); j++){
            double p = responsabilidade(dados.row(i).t(), j);
            if(p > 0)
                entropia_responsabilidades(i) += (p * log2(p));
        }
        entropia_responsabilidades(i) = -entropia_responsabilidades(i);
    }

    for(uword i = 0; i < entropia_responsabilidades.n_elem; i++){
        for(uword j = 0; j < entropia_responsabilidades.n_elem; j++){
            if(entropia_responsabilidades(i) < entropia_responsabilidades(j)){
                entropia_responsabilidades.swap_rows(i,j);
                todos_os_objetos.swap_rows(i,j);
            }
        }
    }
    #if DEBUG_IND
    std::cout << "FEASIBLE -- ORDENOU do menor pro maior\n";
    std::cout << "entropia_responsabilidades: \n" << entropia_responsabilidades << std::endl;
    #endif

    vec probabilidades(dados.n_rows);
    vec intervalos(dados.n_rows);
    double soma = dados.n_rows * (dados.n_rows+1)/2;

    for(uword i = 0; i < dados.n_rows; i++){
        probabilidades[i] = (i+1)/soma;
        intervalos[i] = probabilidades[i];
        if(i > 0)
            intervalos[i] += intervalos[i-1];
    }

    mat retorno(quant_iteracoes, dados.n_cols);

    double num_aleatorio;
    for(uword i = 0; i < quant_iteracoes; i++){
        num_aleatorio = (rand() % 101)/100;
        for(uword j = 0; j < intervalos.n_elem; j++){
            if(num_aleatorio < intervalos[j]){
                for(uword k = 0; k < dados.n_cols; k++){
                    retorno(i, k) = dados(todos_os_objetos(j), k);
                }
                retorno.row(i) = dados.row(todos_os_objetos(j));
                todos_os_objetos.shed_row(j);
                entropia_responsabilidades.shed_row(j);
                probabilidades.shed_row(j);
                intervalos.shed_row(j);
                double soma_probabilidades = sum(probabilidades);
                for(uword k = 0; k < probabilidades.n_elem; k++){
                    probabilidades(k) /= soma_probabilidades;
                    intervalos[k] = probabilidades[k];
                    if(k > 0){
                        intervalos[k] += intervalos[k-1];
                    }
                }
                break;
            }
        }
    }

    return retorno;
}

void Individuo::mutacaoFeasible(uword quant_max_grupos, const mat & dados, std::vector<std::vector<uword>> chunklets){

	double probabilidade, num_aleatorio;
    int quant_min_grupos = chunklets.size();
	probabilidade = ((double)getQuantGrupos() - quant_min_grupos)/(quant_max_grupos - quant_min_grupos);
	num_aleatorio = (rand() % 101)/100;

	if(num_aleatorio < probabilidade){
        vec grupos_a_ser_removidos = selecaoFeasibleEliminacao(quant_min_grupos, dados);
		Operador_Eliminar(grupos_a_ser_removidos);
	}else{
        mat objetos_mut = selecaoFeasibleCriacao(quant_max_grupos, dados);
		Operador_Criar(dados, objetos_mut, chunklets);
	}

    ExpectationMaximization(dados, 2);
}


/**
 * Indica se um grupo é o único mapeado para determinada classe
 * @param matriz de mapeamento de classes e grupos
 * @param índice do grupo a ser verificado
 * @return grupo é o único mapeado para a classe?
 *
 */
bool Individuo::unicoGrupoClasse(uword grupo){
    assert (grupo < getQuantGrupos());
    umat M = getMapeamentoClassesGrupos();
    uword idx_classe = index_max(M.col(grupo));
    return sum(M.row(idx_classe)) == 1;
}

int Individuo::buscaPosicaoDoElementoNoVetor(vec vetor, double elemento){
    uword i;
    assert(false);
	for(i = 0; i < vetor.n_elem; i++){
		if(vetor(i) == elemento){
			return i;
		}
	}
	std::cout << "O elemento " << elemento << " não está no vetor\n";
	return -1;
}

void Individuo::Operador_Eliminar(vec grupos){
    uword i, k;
    grupos = sort(grupos, "descend");
	for(i = 0; i < grupos.n_elem; i++){
		k = grupos(i);
		if(!unicoGrupoClasse(k)){
			removeParametros(k);
			removeLinhaOuColunaDoMapeamentoClassesGrupos('c', k);
			setQuantGrupos(getQuantGrupos() - 1);
		}
	}
    assert(abs(accu(getPesos()) - 1) < 1e-6);
}

void Individuo::Operador_Criar(const mat& dados, mat objetos_mut,std::vector<std::vector<uword>> chunklets){
    assert(abs(accu(getPesos()) - 1) < 1e-6);
    uword j, n, c_n = 0;
	Row<size_t> labels;
	classificaDados(objetos_mut, labels);
#ifdef DEBUG_IND
    std::cout << "CRIAR\nobjetos_mut:\n" << objetos_mut << std::endl;
	std::cout << "labels:\n";
	for(uword i = 0; i < labels.n_elem; i++){
		std::cout << labels(i);
		if(i == labels.n_elem - 1){
			std::cout << std::endl;
		}
		else{
			std::cout << " ";
		}
	}
#endif

	mat mat_cov(getNumAtrib(), getNumAtrib(), fill::zeros);
	vec atrib_covariancia(dados.n_cols);
	for(j = 0; j < dados.n_cols; j++){
		atrib_covariancia(j) = var(dados.col(j));
	}
	mat_cov.diag() += atrib_covariancia * 0.1;

	for(n = 0; n < objetos_mut.n_rows; n++){
		c_n = labels(n);
        vec media = objetos_mut.row(n).t();
        double peso = getPesos(c_n)/2;
		setPesos(peso, c_n);
		adicionaParametros(media, mat_cov, peso);
		//Adicionar K em M, considerando o objeto em um chunklet mais próximo
		adicionaLinhaOuColunaNoMapeamentoClassesGrupos('c', getQuantGrupos() - 1);
        int chunklet_proximo = idxChunkletMaisProximo(chunklets, media.t(), dados);
		setElementoM(chunklet_proximo, getQuantGrupos() - 1, 1);
	}

#ifdef DEBUG_IND
    std::cout << "Ficou com " << getQuantGrupos() << " grupos depois da criação\n";
#endif

    assert(abs(accu(getPesos()) - 1) < 1e-6);
}

vec Individuo::selecaoInfeasibleEliminacao(int quant_min_grupos, int quant_max_grupos, const mat & dados){
    uword quant_iteracoes = quant_max_grupos - quant_min_grupos;
    colvec todos_os_grupos(getQuantGrupos());
    colvec pll_grupos(getQuantGrupos());
    for(uword i = 0; i < getQuantGrupos(); i++){
        todos_os_grupos(i) = i;
        pll_grupos(i) = PartialLogLikelihood(i, dados);
    }

    for(uword i = 0; i < pll_grupos.n_elem; i++){
        for(uword j = 0; j < pll_grupos.n_elem; j++){
            if(pll_grupos(i) > pll_grupos(j)){
                pll_grupos.swap_rows(i,j);
                todos_os_grupos.swap_rows(i,j);
            }
        }
    }
#ifdef DEBUG_IND
    std::cout << "INFEASIBLE -- ORDENOU do maior pro menor\n";
    std::cout << "pll_grupos:\n" << pll_grupos << std::endl;
#endif

    colvec probabilidades(getQuantGrupos());
    colvec intervalos(getQuantGrupos());
    double soma = probabilidades.n_rows*(probabilidades.n_rows+1)/2;

    for(uword i = 0; i < getQuantGrupos(); i++){
        probabilidades[i] = (i+1)/soma;
        intervalos[i] = probabilidades[i];
        if(i > 0)
            intervalos[i] += intervalos[i-1];
    }

#ifdef DEBUG_IND
    std::cout << "probabilidades:\n" << probabilidades << std::endl;
    std::cout << "intervalos:\n" << intervalos << std::endl;
#endif

    vec retorno(quant_iteracoes, fill::zeros);

    double num_aleatorio;
    for(uword i = 0; i < quant_iteracoes; i++){
        num_aleatorio = (rand() % 101)/100;
        for(uword j = 0; j < intervalos.n_elem; j++){
            if(num_aleatorio < intervalos[j]){
                retorno(i) = todos_os_grupos(j);
                todos_os_grupos.shed_row(j);
                pll_grupos.shed_row(j);
                probabilidades.shed_row(j);
                intervalos.shed_row(j);
                double soma_probabilidades = sum(probabilidades);
                for(uword k = 0; k < probabilidades.n_elem; k++){
                    probabilidades(k) /= soma_probabilidades;
                    intervalos[k] = probabilidades[k];
                    if(k > 0){
                        intervalos[k] += intervalos[k-1];
                    }
                }
                break;
            }
        }
    }

    return retorno;
}

std::vector<uword> Individuo::idxObjsViolacao(const mat & dados, std::vector<std::vector<uword>> chunklets){
    Row<size_t> labels;
    classificaDados(dados, labels);
    std::vector<uword> violadores;
    for(uword i = 0; i < chunklets.size(); ++i){
        for(uword j = 0; j < chunklets[i].size(); j++){
            if(!objetoEmGrupoCorreto(i, (int)labels[chunklets[i][j]])){
                 violadores.push_back(chunklets[i][j]);
            }
        }
    }
    if(violadores.size() == 0){
        fprintf(stderr, "\nERRO!\nFunção indiceDeViolacao\nO indivíduo é feasible, logo não possui índice de retorno\n");
        abort();
    }
    return violadores;
}

mat Individuo::selecaoInfeasibleCriacao(int quant_max_grupos, const mat &dados, std::vector<std::vector<uword>> chunklets){
    uword quant_iteracoes = quant_max_grupos - getQuantGrupos();
    std::vector<uword> indices_objetos = idxObjsViolacao(dados, chunklets);
    colvec custo(indices_objetos.size());

    mat todos_os_objetos(indices_objetos.size(), dados.n_cols);

    for(uword i = 0; i < todos_os_objetos.n_rows; i++){
        todos_os_objetos.row(i) = dados.row(indices_objetos[i]);
    }


    Row<size_t> labels;
    classificaDados(todos_os_objetos,labels);


    for(uword i = 0; i < todos_os_objetos.n_rows; i++){
        custo(i) = custoDeViolacaoInfeasible(todos_os_objetos.row(i), (int)labels(i), chunkletDoObj(chunklets, indices_objetos[i]));
    }


    for(uword i = 0; i < custo.n_elem; i++){
        for(uword j = 0; j < custo.n_elem; j++){
            if(custo(i) > custo(j)){
                custo.swap_rows(i,j);
                todos_os_objetos.swap_rows(i,j);
            }
        }
    }

#ifdef DEBUG_IND
    std::cout << "INFEASIBLE -- ORDENOU do menor pro maior\n";
    std::cout << "custo: \n" << custo << std::endl;
#endif

    vec probabilidades(indices_objetos.size());
    vec intervalos(indices_objetos.size());
    double soma = probabilidades.n_rows*(probabilidades.n_rows+1)/2;

    for(uword i = 0; i < indices_objetos.size(); i++){
        probabilidades[i] = (i+1)/soma;
        intervalos[i] = probabilidades[i];
        if(i > 0)
            intervalos[i] += intervalos[i-1];
    }

    mat retorno(quant_iteracoes, dados.n_cols, fill::zeros);

    double num_aleatorio;
    uword i;
    for(i = 0; i < quant_iteracoes && todos_os_objetos.size() != 0; i++){
        num_aleatorio = (rand() % 101)/100;
        for(uword j = 0; j < intervalos.n_elem; j++){
            if(num_aleatorio < intervalos[j]){
                retorno.row(i) = todos_os_objetos.row(j);
                todos_os_objetos.shed_row(j);
                custo.shed_row(j);
                probabilidades.shed_row(j);
                intervalos.shed_row(j);
                double soma_probabilidades = sum(probabilidades);
                for(uword k = 0; k < probabilidades.n_elem; k++){
                    probabilidades(k) /= soma_probabilidades;
                    intervalos[k] = probabilidades[k];
                    if(k > 0){
                        intervalos[k] += intervalos[k-1];
                    }
                }
                break;
            }
        }
    }
    if(i < retorno.n_rows){
        retorno.shed_rows(i,retorno.n_rows-1);
    }
    return retorno;
}

void Individuo::mutacaoInfeasible(uword quant_max_grupos, const mat & dados, std::vector<std::vector<uword>> chunklets){
    int quant_min_grupos = getQuantClasses();

	if(getQuantGrupos() == quant_max_grupos){
        vec grupos_remover = selecaoInfeasibleEliminacao(quant_min_grupos, quant_max_grupos, dados);
		Operador_Eliminar(grupos_remover);
	}
	else{
        mat objetos_mut = selecaoInfeasibleCriacao(quant_max_grupos,
                dados, chunklets);
		Operador_Criar(dados, objetos_mut, chunklets);
	}
    ExpectationMaximization(dados, 2);
}

void Individuo::Imprime(){
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "[" << this << "|prm: " << &(this->getParametros()) << "]" << endl;
    std::cout << "quant_grupos: " << getQuantGrupos() << std::endl;
    std::cout << "quant_classes: " << getQuantClasses() << std::endl;
    std::cout << "num_atrib: " << getNumAtrib() << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "mapeamento_classes_grupos: " <<
        mapeamento_classes_grupos.memptr() << endl <<
        mapeamento_classes_grupos;


    std::cout << "Ponteiros dists" << endl;
    std::cout << getPesos().memptr() << "[" << getPesos().mem << "]" << endl;
    for(uword i = 0; i < getQuantGrupos(); i++){
        std::cout << "[" << i  << "]: Med" << &(getMedia(i)) << endl;
        std::cout << "[" << i  << "]: Med" << (getMedia(i).mem) << endl;
        std::cout << "[" << i  << "]: Cov" << &(getCovariancias(i)) <<endl ;
        std::cout << "[" << i  << "]: Cov" << (getCovariancias(i).mem) <<endl ;
    }

    std::cout << "Média:\n";
    for(uword i = 0; i < getQuantGrupos(); i++){
        std::cout << "[" << i  << "]:" << getMedia(i).t();
    }
    std::cout << std::endl;
    std::cout << "Covariâncias:\n";
    for(uword i = 0; i < getQuantGrupos(); i++){
        std::cout << "  slice[" << i << "]:" << std::endl;
        std::cout << getCovariancias(i).t() << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Pesos: " << accu(getPesos()) << endl;
    for(uword i = 0; i < getQuantGrupos(); i++){
        std::cout << "[" << i  << "]:" << getPesos(i);
    }
    std::cout << std::endl << "-----------------------------------" << std::endl;
}

