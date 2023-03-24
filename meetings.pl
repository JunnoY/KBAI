nonmember_of(_X,[]). % Any element is not in the empty list
nonmember_of(X,[Head|Tail]) :- dif(X,Head), nonmember_of(X, Tail). % X is different from the head, % X is not in the tail

alldifferent([], []).
alldifferent([Head|Tail], X) :- nonmember_of(Head, X), alldifferent(Tail, X); nonmember_of(Head, X),=(Tail,[]) ; nonmember_of(Head, X),=(X,[]) .

% schedule exercise
meetings_one_two_three(A-B,C-D,E-F) :-
%    student(A),
%    student(B),
%    student(C),
%    student(D),
%    student(E),
%    student(F),
%    (slot1 = A-_ :- A = student1),
%    (not(S = A-_, not(S = slot1)) :- A = student1),
%    ((S = B-C) :- B = student2, C = student3),
%    (not(Y = B-_, Y = C-_, not(X=Y)) :- B = student2, C = student3),
%    (X = A-_, Y = D-_, not(X=Y) :- A = student1, D = student4),
    (not(X = A-_, Y = D-_, X=Y) :- B = student2, C = student3, not(X = F-_)).


