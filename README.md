# COMP24412 Labs

This is the base repo for COMP24412 labs in the academic session 2022-23.

This branch holds the materials for the `lab1` assignment.
You can always return to this branch with the command
```
git checkout lab1
```

There is a refresh script to fetch the lab materials when they become available.
You **must** run this script before you start working on the assignment.
This can be done with the command
```
./refresh.sh
```

To submit your work you need to follow the coursework instructions in the [CS Handbook](https://wiki.cs.manchester.ac.uk/index.php/UGHandbook22:Coursework#Developing_and_submitting_with_Gitlab).
You **must** use the correct git tag which you can find in the manual for the lab.
This usually involves the following sequence of commands
```
git add -A .
git commit
git tag lab1_solution
git push origin
git push --tags origin
```

The [CS Handbook](https://wiki.cs.manchester.ac.uk/index.php/UGHandbook22:Coursework#Correcting_an_Incorrect_Tag) also has instructions in case you need to ammend your submission and then fix your submission tag.

Please ask for support in the lab sessions if you're unsure about lab submission and instructions.
