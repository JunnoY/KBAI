include('SET001-0.ax').

fof(a, axiom, ![Y]:(member(Y,dirimg(A))<=>?[X]: (member(X,A) & role(X,Y)))).

fof(b, axiom, ![X]:(member(X,valres(B))<=>![Y]: (role(X,Y) => member(Y,B)))).

fof(c, conjecture, ![A,B]:((subset(dirimg(A),B) | equal_sets(dirimg(A),B))<=>(subset(A,valres(B)) | equal_sets(A,valres(B))))).