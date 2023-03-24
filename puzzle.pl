side_opposite(north,south).
side_opposite(south,north).

% wolf, goat, cabbage, nothing are moves
move([X,X,Goat,Cabbage], wolf, [Y,Y,Goat,Cabbage]) :- side_opposite(X,Y). % now move farmer and wolf, goat and cabbage can be at any direction
move([X,Wolf,X,Cabbage], goat, [Y,Wolf,Y,Cabbage]) :- side_opposite(X,Y).
move([X,Wolf,Goat,X], cabbage, [Y,Wolf,Goat,Y]) :- side_opposite(X,Y).
move([X,Wolf,Goat,Cabbage], nothing, [Y,Wolf,Goat,Cabbage]) :- side_opposite(X,Y).

safety_check(X,X,_). % either the first and the second element at the same place
safety_check(X,_,X). % or the first and the third element at the same place

safe_state([Farmer,Wolf,Goat,Cabbage]) :- safety_check(Farmer,Goat,Wolf), safety_check(Farmer,Goat,Cabbage).

solution([south,south,south,south],[]). % initial state
solution(CurrentState, [Move|OtherMoves]) :- move(CurrentState,Move,NextState), safe_state(NextState), solution(NextState,OtherMoves).

% length(X,7), solution([north,north,north,north],X). # set the solution to be a list of length 7
