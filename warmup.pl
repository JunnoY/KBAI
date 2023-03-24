loves(monkey,banana).
loves(mouse,cheese).
loves(dad,mum,kid).

isa_list([]).
isa_list([Head|Tail]) :- isa_list(Tail).

nonmember_of(_X,[]). % Any element is not in the empty list
nonmember_of(X,[Head|Tail]) :- dif(X,Head), nonmember_of(X, Tail). % X is different from the head, % X is not in the tail

bigger_than_one([_,_|_]).

same_head([X|_],[X|_]).

same_sec_head([_,X|_],[_,X|_]).

related([Head|Tail], [Head1|Tail1]) :- =(Head,Head1), related(Tail,Tail1); =(Head,Head1), =(Tail,[]) ; =(Head,Head1), =(Tail1,[]).
