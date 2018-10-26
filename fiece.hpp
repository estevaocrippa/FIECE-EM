#include "populacao.hpp"

struct Aresta {
    int origem, destino; //origem e destino da aresta direcionada
};
typedef struct Aresta aresta;

struct Grafo {
    int V; // Numero de vertices
    int E; // Numero de arestas

    // O grafo eh representado por um vetor de arestas
    aresta* arestas;
};
typedef struct Grafo grafo;

struct Subset {
    int pai;
    int rank;
};
typedef struct Subset subset;


typedef struct fiecem_prm{
    uword max_geracoes;
    uword max_grupos;
    uword max_tentativas;
    uword tam_populacao;
    uword num_em_it;
}  fiecem_prm;

grafo* criarGrafo(arma::mat, int); //cria um grafo com um dado numero de vertices
//(primeiro parametro) e um dado numero de arestas (segundo parametro)
int Find_Set(subset*, int); //funcao que procura o
//representante do elemento i em um grafo
void Union(subset*, int, int); //une dois conjuntos distintos
//x e y
arma::vec ComponentesConexas(grafo*);
arma::vec objetosRepresentantesDeObjetos(arma::mat, arma::mat);
std::vector<std::vector<arma::uword>> geraChunklets(arma::mat, arma::uword);
//retorna um vetor, com o tamanho igual ao numero de objetos,
//que contem os indices dos objetos representantes de cada objeto dentro
//de um chunklet
arma::vec indicesDosObjetosRepresentantesDosChunklets(arma::mat, arma::mat);
arma::vec idxRepresChunklets(arma::vec);
//adjacente criada a partir da funcao restricoesToMatrizAdj-1.
void geraPopulacoes(Populacao&, Populacao&, fiecem_prm, std::vector<std::vector<arma::uword>>, const arma::mat&);
Individuo* fieceem(const arma::mat &, const arma::mat &, const fiecem_prm parametros);

