fof(distinct_rooms, axiom, $distinct(r1,r2,r3,r4,r5,r6)).

fof(order_of_ground_rooms, axiom, next(r1,r2) & next(r2,r3) & next(r3,r4) & next(r4,r5) & next(r5,r6)
    & ~(next(r1,r1) | next(r1,r3) | next(r1,r4) | next(r1,r5) | next(r1,r6))
    & ~(next(r2,r1) | next(r2,r2) | next(r2,r4) | next(r2,r5) | next(r2,r6))
    & ~(next(r3,r1) | next(r3,r2) | next(r3,r3) | next(r3,r5) | next(r3,r6))
    & ~(next(r4,r1) | next(r4,r2) | next(r4,r3) | next(r4,r4) | next(r4,r6))
    & ~(next(r5,r1) | next(r5,r2) | next(r5,r3) | next(r5,r4) | next(r5,r5))
    & ~(next(r6,r1) | next(r6,r2) | next(r6,r3) | next(r6,r4) | next(r6,r5) | next(r6,r6))
).

fof(only_one_animal_in_one_room, axiom,
    ~((dog(r1) & cat(r1)) | (dog(r1) & hamster(r1)) | (cat(r1) & hamster(r1)) | (dog(r1) & cat(r1) & hamster(r1)))
  & ~((dog(r2) & cat(r2)) | (dog(r2) & hamster(r2)) | (cat(r2) & hamster(r2)) | (dog(r1) & cat(r2) & hamster(r2)))
  & ~((dog(r3) & cat(r3)) | (dog(r3) & hamster(r3)) | (cat(r3) & hamster(r3)) | (dog(r3) & cat(r3) & hamster(r3)))
  & ~((dog(r4) & cat(r4)) | (dog(r4) & hamster(r4)) | (cat(r4) & hamster(r4)) | (dog(r4) & cat(r4) & hamster(r4)))
  & ~((dog(r5) & cat(r5)) | (dog(r5) & hamster(r5)) | (cat(r5) & hamster(r5)) | (dog(r5) & cat(r5) & hamster(r5)))
  & ~((dog(r6) & cat(r6)) | (dog(r6) & hamster(r6)) | (cat(r6) & hamster(r6)) | (dog(r6) & cat(r6) & hamster(r6)))
).

fof(r6_hamster_no_light, axiom, hamster(r6) & ~lit(r6)).

fof(room_occupied_by_dog_or_cat, axiom, (~(dog(r1)<=>cat(r1)) & ~(dog(r2)<=>cat(r2)) & ~(dog(r3)<=>cat(r3)) & ~(dog(r4)<=>cat(r4)) & ~(dog(r5)<=>cat(r5)))).

fof(nervous_condition_for_dog, axiom, ![V] : ((dog(V) & ?[U,W]: (dog(U) & dog(W) & next(U,V) & next(V,W)))=> lit(V))).

fof(nervous_condition_for_cat, axiom, ![V] : ((cat(V) & ?[U]: (cat(U) & (~(next(U,V) <=> next(V,U))))) => (lit(V) & lit(U)))).

fof(lit_implication_dog, axiom, ![V] : ((lit(V) & dog(V)) => ?[U,W] : (dog(U) & dog(W) & next(U,V) & next(V,W)))).

fof(lit_implication_cat, axiom, ![V] : ((lit(V) & cat(V)) => ?[U] : (cat(U) & (~(next(U,V) <=> next(V,U))) & lit(U)))).


fof(lit_rooms, axiom,
    (lit(r1)|lit(r2)|lit(r3)|lit(r4)|lit(r5)|lit(r6))
    &
    (
        (~lit(r1)|~lit(r2))&
        (~lit(r1)|~lit(r3))&
        (~lit(r1)|~lit(r4))&
        (~lit(r1)|~lit(r5))&
        (~lit(r1)|~lit(r6))&
        (~lit(r2)|~lit(r3))&
        (~lit(r2)|~lit(r4))&
        (~lit(r2)|~lit(r5))&
        (~lit(r2)|~lit(r6))&
        (~lit(r3)|~lit(r4))&
        (~lit(r3)|~lit(r5))&
        (~lit(r3)|~lit(r6))&
        (~lit(r4)|~lit(r5))&
        (~lit(r4)|~lit(r6))&
        (~lit(r5)|~lit(r6))
    )
).