% The first section defines the components of the system: antenna, transponder, radar, spectrometer, IMU, camera, CPU,
% and RAM. These are basic building blocks that can be used to construct designs.
component(antenna).
component(transponder).
component(radar).
component(spectrometer).
component(imu).
component(camera).
component(cpu).
component(ram).

% The safe_with/2 predicate defines the safety relationships between components. For example, safe_with(radar,cpu) means
% that the radar and CPU are safe to use together.
safe_with(radar,cpu).
safe_with(cpu,radar).
safe_with(radar,imu).
safe_with(imu,radar).
safe_with(imu,camera).
safe_with(camera,imu).
safe_with(imu,cpu).
safe_with(cpu,imu).
safe_with(imu,ram).
safe_with(ram,imu).
safe_with(ram,cpu).
safe_with(cpu,ram).
safe_with(ram,camera).
safe_with(camera,ram).
safe_with(camera,transponder).
safe_with(transponder,camera).
safe_with(camera,cpu).
safe_with(cpu,camera).
safe_with(cpu,spectrometer).
safe_with(spectrometer,cpu).
safe_with(cpu,antenna).
safe_with(antenna,cpu).
safe_with(antenna,spectrometer).
safe_with(spectrometer,antenna).
safe_with(antenna,transponder).
safe_with(transponder,antenna).
safe_with(transponder,spectrometer).
safe_with(spectrometer,transponder).


% This experiment uses unification, where:
% [part(C)|Tail] already restrict that the input has to be a valid design
% [shield([part(C)|T] already restrict that the input has to be a valid design

% Exercise 1

% The safe_list/1 predicate checks if a list of components is safe to use together based on their safe_with
% relationships.
safe_list([]).
safe_list([Head]) :-
    component(Head).
safe_list([Head1, Head2|Tail]) :-
    safe_with(Head1, Head2),
    safe_list([Head1|Tail]),
    safe_list([Head2|Tail]).

% The design/1 predicate defines a valid design, which can be a list of components or a shield made up of a list of
% components.
design([]).
design([part(C)|Tail]) :- component(C), design(Tail).
design([shield([part(C)|T])|Tail]) :- design([part(C)|T]), design(Tail).

% Exercise 2

% The safe_design/1 predicate checks if a design is safe. It first checks if the design is valid, then checks if the
% components in the design are safe to use together.
safe_design([]) :- design([]).

safe_design([part(A)]) :-
    design([part(A)]),
    safe_list([A]).

safe_design([shield([part(C)|T])]) :-
    design([shield([part(C)|T])]),
    safe_design([part(C)|T]).

safe_design([part(A),part(B)|Tail]) :-
    design([part(A),part(B)|Tail]),
    safe_list([A,B]),
    safe_design([part(A)|Tail]),
    safe_design([part(B)|Tail]).

safe_design([shield([part(C)|T]),shield([part(C2)|T2])|Tail]) :-
    design([shield([part(C)|T]),
    shield([part(C2)|T2])|Tail]),
    safe_design([part(C)|T]),
    safe_design([part(C2)|T2]),
    safe_design(Tail).

safe_design([part(A),shield([part(C)|T])|Tail]) :-
    design([part(A),shield([part(C)|T])|Tail]),
    safe_design([part(C)|T]),
    safe_list([A]),
    safe_design([part(A)|Tail]).

safe_design([shield([part(C)|T]),part(B)|Tail]) :-
    design([shield([part(C)|T]),part(B)|Tail]),
    safe_design([part(C)|T]), safe_list([B]),
    safe_design([part(B)|Tail]).


% Exercise 3

% The count_shields/2 predicate counts the number of shields in a design.
% Have to check if the input list is a design (or should we check if it is safe_design)
count_shields([],0).

% [shield([part(C)|T] already restrict that the input has to be a valid design
count_shields([shield([part(C)|T])|Tail],N) :-
    count_shields(Tail, Count1),
    count_shields([part(C)|T], Count2),
    N is Count1 + Count2 + 1.

% [part(C)|Tail] already restrict that the input has to be a valid design
count_shields([part(_)|Tail],N) :- count_shields(Tail,N).

% Exercise 4 & 5
% split_list/3 that takes a list as its first argument and returns two lists as its second and third arguments, where
% the first list contains the first half of the original list (or the middle element if the length of the list is odd)
% and the second list contains the second half of the original list.

% base case for an empty list, where both the second and third arguments are empty lists.
split_list([],[],[]).

% Recursively calling split_list/3 on the tail of the list. Adds the head of the original list to the first argument of
% the resulting split
split_list([X|Xs],[X|Ys],Zs) :- split_list(Xs,Ys,Zs).

% Recursively calling split_list/3 on the tail of the list. Adds the head of the original list to the second argument
% of the resulting split.
split_list([X|Xs],Ys,[X|Zs]) :- split_list(Xs,Ys,Zs).

% The design_uses/2 predicate checks if a list of components is a proper subset of a design and if the design is valid.

% The implementation of design_uses will not go into infinite loop because:
% 1. here is a base case in the first line design_uses([],[]) :- design([]).
% which provides a termination condition for the recursion.
% Also, within design_uses I used split_list and design, and both of them also have a base case which provides a
% termination condition for the recursion.
% 2. design is run at the end of design_uses on purpose because design does not consider the component list, it can
% generate infinite designs that does not satisfy the component list and if we try to all designs first, then our
% program will get into infinite loop. For this reason, we run split_list first because we list of components will
% always be finite and in all tests the finite component list will always be given, so we can recursively call
% split_list and design_uses to generate the finite possible solutions first then test them using design.
% 3. This experiment uses unification, where:
% [part(C)|Tail] already restrict that the input has to be a valid design
% [shield([part(C)|T] already restrict that the input has to be a valid design
% Therefore, design_uses will not go into infinite loop
design_uses([],[]) :- design([]).

design_uses([part(C)|Tail],List) :-
% Select a component C from the design and extract it from the component list
    split_list(List,[C],T),
    design_uses(Tail,T),
    design([part(C)|Tail]).

design_uses([shield([part(C)|T])|Tail],List) :-
    split_list(List,[C],T3), % T3 is the remaining list after extracting component C from List
    % Telling prolog T3 is split into T1 and T2, which then T1 is used with T, and T2 is used with Tail
    split_list(T3,T1,T2),
    design_uses(T,T1),
    design_uses(Tail,T2),
    design([shield([part(C)|T])|Tail]).