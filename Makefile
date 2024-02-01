.PHONY: clean-project archive

clean-project:
    cd mypkg
	make clean
	cd ..

archive: clean-project
    ifndef NAME
        $(error NAME is not set. Usage: make archive NAME=<project_name>)
    endif
    # Temporarily replace 'mypkg' with the new project name in all files
    find $(CURDIR)/mypkg -type f -exec sed -i 's/mypkg/$(NAME)/g' {} +

    # Temporarily replace 'mypkg' with the new project name in all directories
	find $(CURDIR) -depth -type d -name '*mypkg*' | while read dir; do mv "$dir" "$(dirname "$dir")/$(basename "$dir" | sed 's/mypkg/$(NAME)/g')"; done
    
    # Create the zip file
    zip -r $(NAME).zip $(CURDIR) -x "$(CURDIR)/.git/*" "$(CURDIR)/.vscode/*"
    
    # Revert the changes by replacing the new project name back with 'mypkg'
    find $(CURDIR) -type f -exec sed -i 's/$(NAME)/mypkg/g' {} +

    # Revert the changes by replacing the new project name back with 'mypkg' in all directories
	find $(CURDIR) -depth -type d -name '*$(NAME)*' | while read dir; do mv "$dir" "$(dirname "$dir")/$(basename "$dir" | sed 's/$(NAME)/mypkg/g')"; done
