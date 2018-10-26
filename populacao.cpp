#include "populacao.hpp"

//#define DEBUG_POP 1
Populacao::Populacao()
    : max_individuos(0),
      num_individuos(0),
      individuos() { }

/**
 * Construtor padrão
 *
 * @param tam_populacao quantidade de indivíduos
 */
Populacao::Populacao(int tam_populacao)
    : max_individuos(tam_populacao),
      num_individuos(0),
      individuos() {
    this->max_individuos = tam_populacao;
    this->num_individuos = 0;
    this->individuos = new Individuo*[this->max_individuos];
    for(int i=0; i < tam_populacao;++i)
        this->individuos[i] = NULL;
}

Populacao::Populacao(const Populacao& outro)
    : max_individuos(outro.getMaxIndividuos()),
      num_individuos(outro.getNumIndividuos()),
      individuos(){
    this->individuos = new Individuo*[this->max_individuos];
    for(uword i=0; i<num_individuos; ++i){
        this->individuos[i] = new Individuo(outro.getIndividuo(i));
    }
}


Populacao& Populacao::operator=(const Populacao& outro){
    max_individuos = outro.getMaxIndividuos();
    num_individuos = outro.getNumIndividuos();
    individuos = new Individuo*[this->max_individuos];
    for(uword i=0; i<num_individuos; ++i){
        this->individuos[i] = new Individuo(outro.getIndividuo(i));
    }
    return *this;
}

Populacao::~Populacao(){
    for(uword i = 0; i < this->num_individuos; i++){
        delete individuos[i];
    }
    delete[] this->individuos;
}


uword Populacao::getMaxIndividuos() const{
    return this->max_individuos;
}

uword Populacao::getNumIndividuos() const{
    return this->num_individuos;
}

Individuo** Populacao::getIndividuos() const{
    return this->individuos;
}

Individuo* Populacao::getIndividuo(int i) const{
    return this->individuos[i];
}

void Populacao::trocaIndividuo(uword i, uword j){
    Individuo* aux = getIndividuo(i);
    this->individuos[i] = this->individuos[j];
    this->individuos[j] = aux;
}

void Populacao::setIndividuo(uword i, Individuo* individuo){
    if(i < max_individuos){
        delete this->individuos[i];
        this->individuos[i] = individuo;
    }else{
        fprintf(stderr, "\nERRO!\nFunção setIndividuo\nPosição extrapola o limite da quantidade de indivíduos\n");
    }
}

void Populacao::bubblesortFeasible(const mat & dados){
    for(uword i = 0; i < getNumIndividuos(); i++){
        double f1 = getIndividuo(i)->funcaoObjetivoFeasible(dados);
        for(uword j = i+1; j < getNumIndividuos(); j++){
            double f2 = getIndividuo(j)->funcaoObjetivoFeasible(dados);
            if(f1 < f2){
                trocaIndividuo(i, j);
                f1 = f2;
            }
        }
    }
}


void Populacao::selecaoMiMaisLambda(const Populacao& feasible_pool, const arma::mat & dados, std::vector<std::vector<arma::uword>> chunklets){
    uword ttl_indiv = getNumIndividuos() + feasible_pool.getNumIndividuos();
    Populacao todos_os_individuos(ttl_indiv);

#ifdef DEBUG_POP
    cout << "SelecaoMiMaisLambda antes de sort, juntando: " << endl;
    ImprimeVetor_individuos();
    cout << "   com: " << endl;
    feasible_pool.ImprimeVetor_individuos();
#endif

    for(uword i = 0; i < ttl_indiv; i++){
        if(i < getNumIndividuos()){
            todos_os_individuos.adicionarIndividuo(new Individuo(getIndividuo(i)));
        }else{
           todos_os_individuos.adicionarIndividuo(new Individuo(feasible_pool.getIndividuo(i-getNumIndividuos())));
        }
    }

#ifdef DEBUG_POP
    cout << "SelecaoMiMaisLambda antes de sort, todos os indivíduos" << endl;
    todos_os_individuos.ImprimeVetor_individuos();
#endif

    todos_os_individuos.bubblesortFeasible(dados);

    uword i;
    for(i = 0; i < getMaxIndividuos() && i < ttl_indiv; i++){
        setIndividuo(i, new Individuo(todos_os_individuos.getIndividuo(i)));
        //this->individuos[i] = new Individuo(todos_os_individuos.getIndividuo(i));
    }
    num_individuos = i;
    for(; i<getMaxIndividuos();++i){
        delete this->individuos[i];
    }


#ifdef DEBUG_POP
    cout << "SelecaoMiMaisLambda após sort, todos os indivíduos" << endl;
    todos_os_individuos.ImprimeVetor_individuos();
#endif


}


void Populacao::bubblesortInfeasible(std::vector<std::vector<arma::uword>> chunklets, const arma::mat& dados){
    for(uword i = 0; i < getNumIndividuos(); i++){
        double f1 = getIndividuo(i)->funcaoObjetivoInfeasible(chunklets, dados);
        for(uword j = i+1; j < getNumIndividuos(); j++){
            double f2 = getIndividuo(j)->funcaoObjetivoInfeasible(chunklets, dados);
            if(f1 < f2){
                trocaIndividuo(i, j);
                f1 = f2;
            }
        }
    }
}

void Populacao::selecaoRoleta(const Populacao& infeasible_pool, const arma::mat &dados, std::vector<std::vector<arma::uword>> chunklets){
    uword ttl_indiv = getNumIndividuos() + infeasible_pool.getNumIndividuos();
    Populacao todos_os_individuos(ttl_indiv);
    for(uword i = 0; i < ttl_indiv; i++){
        if(i < getNumIndividuos()){
            todos_os_individuos.adicionarIndividuo(new Individuo(getIndividuo(i)));
        }
        else{
            todos_os_individuos.adicionarIndividuo(new Individuo(infeasible_pool.getIndividuo(i - getNumIndividuos())));
        }
    }


#ifdef DEBUG_POP
    cout << "SelecaoRoleta antes sort, todos os indivíduos" << endl;
    todos_os_individuos.ImprimeVetor_individuos();
#endif

    todos_os_individuos.bubblesortInfeasible(chunklets, dados);

#ifdef DEBUG_POP
    cout << "SelecaoRoleta após sort, todos os indivíduos" << endl;
    todos_os_individuos.ImprimeVetor_individuos();
#endif

    arma::vec probabilidades(todos_os_individuos.getNumIndividuos());
    arma::vec intervalos(todos_os_individuos.getNumIndividuos());
    double soma = ttl_indiv*(ttl_indiv+1)/2.0;

    for(uword i = 0; i < ttl_indiv; i++){
        probabilidades[i] = (todos_os_individuos.getNumIndividuos() - i)/soma;
        intervalos[i] = probabilidades[i];
        if(i > 0)
            intervalos[i] += intervalos[i-1];
    }


    /**
     * Atualmente fazendo mais cópias do que deveria para remover o indivíduo
     * TODO remover cópias mantendo index indireto
     */
    double num_aleatorio;
    uword i;
    for(i = 0; i < getMaxIndividuos() && i < ttl_indiv; i++){
        num_aleatorio = (rand() % 101)/100;
        for(uword j = 0; j < intervalos.n_elem; j++){
            if(num_aleatorio < intervalos[j]){
                setIndividuo(i, new Individuo(todos_os_individuos.getIndividuo(j)));
                todos_os_individuos.removerIndividuo(j);
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
    num_individuos = i;
    for(; i < getMaxIndividuos(); i++)
        delete this->individuos[i];
}

void Populacao::ImprimeVetor_individuos() const{
    for(uword i = 0; i < getNumIndividuos(); i++){
        std::cout << "INDIVIDUO " << i << "\n-----------------------------------\n";
        std::cout << "[" << this << "|ind: " << individuos << "]" << endl;
        getIndividuo(i)->Imprime();
    }
}

bool Populacao::adicionarIndividuo(Individuo* novo_individuo){
    if(this->num_individuos == this->max_individuos){
#ifdef DEBUG_POP
        cout << "Populacao cheia" << novo_individuo << endl;
#endif
        return false;
    }
    this->individuos[this->num_individuos++] = novo_individuo;
    return true;
}

void Populacao::removerIndividuo(uword posicao){
    Individuo** vetor_copia = new Individuo*[getNumIndividuos() - 1];
    if(posicao < getNumIndividuos()){
        uword j = 0;
        for(uword i = 0; i < getNumIndividuos(); i++){
            if(i == posicao){
                delete this->individuos[i];
                continue;
            }
            vetor_copia[j] = this->individuos[i];
            j++;
        }
        delete[] this->individuos;

        num_individuos--;
        this->individuos = vetor_copia;
    }
    else{
        fprintf(stderr, "\nERRO!\nFunção removerIndividuo\nPosição extrapola o limite da quantidade de indivíduos\n");
    }
}

Individuo* Populacao::melhorFeasible(const mat& dados){
    if(num_individuos == 0) return NULL;
    double melhor = individuos[0]->funcaoObjetivoFeasible(dados), fun;
    uword idx = 0;
    for(uword i=1; i < num_individuos; ++i){
        fun = individuos[i]->funcaoObjetivoFeasible(dados);
        if(fun > melhor){
            idx = i;
            melhor = fun;
        }
    }
    return individuos[idx];
}

Individuo* Populacao::melhorInfeasible(std::vector<std::vector<uword>> chunklets, const mat& dados){
    if(num_individuos == 0) return NULL;
    double melhor = individuos[0]->funcaoObjetivoInfeasible(chunklets, dados), fun;
    uword idx = 0;
    for(uword i=1; i < num_individuos; ++i){
        fun = individuos[i]->funcaoObjetivoInfeasible(chunklets, dados);
        if(fun < melhor){
            idx = i;
            melhor = fun;
        }
    }
    return individuos[idx];
}

