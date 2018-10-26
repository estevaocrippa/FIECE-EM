CFLAGS = -fopenmp -larmadillo -lmlpack -lboost_serialization -lm -Wall -ggdb
COVFLAGS = -fprofile-arcs -ftest-coverage -fPIC
all:	testes#main.exe

individuo.o: individuo.cpp individuo.hpp
	g++ -c individuo.cpp $(CFLAGS) $(COVFLAGS)

populacao.o: populacao.cpp populacao.hpp
	g++ -c  populacao.cpp $(CFLAGS) $(COVFLAGS)

fiece.o: fiece.cpp fiece.hpp
	g++ -c  fiece.cpp $(CFLAGS) $(COVFLAGS)

main.o:	main.cpp
	g++ -c  main.cpp

main.exe: individuo.o populacao.o fiece.o main.o
	g++ -o individuo.o populacao.o fiece.o main.o -larmadillo -lmlpack -lboost_serialization -lm

clean:
	rm individuo.o populacao.o fiece.o main.o

roda_exp: fiece.o populacao.o individuo.o
	g++ -o roda_exp roda_exp.cpp fiece.o populacao.o individuo.o $(CFLAGS) $(COVFLAGS)


