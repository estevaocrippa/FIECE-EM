#pragma once
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mlpack/methods/gmm/gmm.hpp>
#include <mlpack/core/dists/gaussian_distribution.hpp>

using namespace mlpack::gmm;
using namespace arma;


/*
 * Classe de individuo; cada individuo representa uma solucao.
 * Parametros:
 * - quant_classes: quantidade de classes/clusters;
 * - quant_grupos: quantidade de grupos;
 * - mapeamento_classes_grupos: matriz quant_classes x quant_grupos
 * que faz a relacao entre clusters e grupos.
 * O elemento recebe 1 caso o grupo esteja associado ao cluster e,
 * 0, caso contrario;
 * - parametros: variavel do tipo GMM que guarda um modelo
 * de mistura, i.e., as medias, covariancias e pesos;
 */
class Individuo{
    private:
        uword quant_classes, quant_grupos, num_atrib;
        umat mapeamento_classes_grupos;
        GMM parametros;
    public:
        Individuo(); //funcao construtora de um individuo
        Individuo(Individuo *);//função copia
        Individuo(uword, std::vector<std::vector<uword>>, const mat&);
        Individuo(int, int, int, vec*, mat*, vec);
        //funcao construtora; recebe quantidade de classes,
        //grupos, numero de atributos, ponteiro pra vetor de medias,
        //ponteiro pra matriz de covariancia,
        //vetor de pesos como parametros
        ~Individuo(); //funcao destrutora
        void setQuantClasses(uword ); //muda a quantidade de classes
        uword getQuantClasses(); //retorna a quantidade de classes
        void setQuantGrupos(uword ); //muda a quantidade de grupos
        uword getQuantGrupos(); //retorna a quantidade de grupos
        void setNumAtrib(uword ); //altera a quantidade de atributos
        uword getNumAtrib(); //retorna a quantidade de atributos
        void setPesos(vec); //muda os pesos dado um vetor
        //que contem os pesos de cada componente
        void setPesos(double, uword); //muda os pesos; usa o novo peso e a
        //posicao como parametros
        double getPesos(int); //retorna o peso na posicao passada como
        //parametro
        vec& getPesos(); //retorna um vetor que contém os pesos
        //de cada componente
        void setMapeamentoClassesGrupos(umat);
        const umat& getMapeamentoClassesGrupos(); //retorna a matriz que faz o mapeamento
        //entre classes e grupos
        uword getMapeamentoClassesGrupos(int, int);
        void removeLinhaOuColunaDoMapeamentoClassesGrupos(char, uword); //remove a linha, se o primeiro
        //parametro for 1, ou a coluna, se for 2. O segundo parametro
        //indica a linha/coluna removida.
        void adicionaLinhaOuColunaNoMapeamentoClassesGrupos(char, uword);
        void setElementoM(uword, uword, uword);
        const GMM& getParametros(); //retorna um parametro do tipo GMM
        vec* getMedias();
        const vec& getMedia(int);
        mat* getCovariancias();
        const mat& getCovariancias(int);
        void adicionaParametros(vec, mat, double); //dado um vetor de
        //medias e uma matriz de covariancia, altera os dados parametros
        //do individuo numa posicao passada como parametro da funcao
        void removeParametros(uword); //remove os parametros do individuo
        //numa dada posicao passada como parametro para a funcao
        umat geraCentroides(mat&, const mat&, uword, std::vector<std::vector<uword>>);
        void criaMapeamentoClassesGrupos(); //altera o tamanho da matriz M;
        //recebe quantidade de classes e de grupos como
        //parametros, respectivamente
        mat mapeiaObjetosEGrupos_matriz(mat, int);
        void classificaDados(const mat&, Row<size_t>&);
        //dada uma matriz de dados, classifica cada elemento como pertencente
        //a um dado grupo e insere as componentes no vetor passado como
        //parametro da funcao
        double getProbabilidadeGMM(vec, int);
        bool objetoEmGrupoCorreto(int, int);
        double custoDeViolacaoInfeasible(rowvec, int, int);
        bool ehFeasible(mat, std::vector<std::vector<uword>>);
        double funcaoObjetivoInfeasible(std::vector<std::vector<uword>> chunklets, const mat& dados);
        double elementoAuxiliar(int, mat);
        double responsabilidade(const vec &, uword);
        double Likelihood(const mat&);
        double funcaoObjetivoFeasible(const mat &);
        void removeGruposVazios(mat);
        bool objetoEstaNosChunklets(std::vector<std::vector<uword>>, uword);
        int chunkletDoObj(std::vector<std::vector<uword>>, uword);
        int idxChunkletMaisProximo(std::vector<std::vector<uword>>, rowvec, const mat &);
        void zeraColunaESetElementoM(int, int, int);
        void rotularGruposComObjetosForaDeChunklets(std::vector<std::vector<uword>>, const mat &);
        double ExpectationMaximization(const mat &, int);
        double PartialLogLikelihood(uword, const mat &);
        vec selecaoFeasibleEliminacao(int, const mat &);
        mat selecaoFeasibleCriacao(int, const mat &);
        void mutacaoFeasible(uword,const mat &, std::vector<std::vector<uword>>);
        bool ehGrupoVazio(mat, int, int);
        bool unicoGrupoClasse(uword);
        int buscaPosicaoDoElementoNoVetor(vec, double);
        void Operador_Eliminar(vec);
        void Operador_Criar(const mat &, mat, std::vector<std::vector<uword>>);
        vec selecaoInfeasibleEliminacao(int, int, const mat &);
        std::vector<uword> idxObjsViolacao(const mat&, std::vector<std::vector<uword>>);
        mat selecaoInfeasibleCriacao(int, const mat &, std::vector<std::vector<uword>>);
        void mutacaoInfeasible(uword, const mat &, std::vector<std::vector<uword>>);
        void Imprime(); //imprime todos os parametros
        //contidos no Individuo
};

