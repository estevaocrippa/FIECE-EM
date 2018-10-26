#pragma once
#include "individuo.hpp"

using namespace arma;
/*
 * Classe de populacao
 * Parametros:
 * - num_individuos: representa a quantidade de individuos que a populacao possui;
 * - individuos: contem os individuos que formam a populacao;
 */
class Populacao{
    private:
        uword max_individuos;
        uword num_individuos;
        Individuo** individuos;
    public:
        Populacao(); //funcao construtora de Populacao
        Populacao(int); //funcao construtora; recebe a quantidade de individuos da populacao
        Populacao(const Populacao&);
        ~Populacao(); //funcao destrutora
        Populacao& operator=(const Populacao&);
        void setNumIndividuos(int); //muda a quantidade de individuos
        uword getNumIndividuos() const; //retorna o numero de individuos
        uword getMaxIndividuos() const; //retorna o numero de individuos
        Individuo** getIndividuos() const;
        Individuo* getIndividuo(int) const;
        void setIndividuo(uword, Individuo*);
        void bubblesortFeasible(const mat &);
        void selecaoMiMaisLambda(const Populacao&,const arma::mat &, std::vector<std::vector<arma::uword>>);
        void bubblesortInfeasible(std::vector<std::vector<arma::uword>> chunklets, const arma::mat& dados);
        void selecaoRoleta(const Populacao&, const arma::mat &, std::vector<std::vector<arma::uword>>);
        void ImprimeVetor_individuos() const; //imprime o vetor de individuos
        bool adicionarIndividuo(Individuo*); //adiciona um individuo, como parametro, na populacao
        void removerIndividuo(uword);
        Individuo *melhorFeasible(const mat&);
        Individuo *melhorInfeasible(std::vector<std::vector<arma::uword>>, const mat&);
        void trocaIndividuo(uword, uword);

};

