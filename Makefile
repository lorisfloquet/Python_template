.PHONY: clean-project create

clean-project:
	@cd mypkg && make clean && cd ..

create: clean-project
ifndef NAME
	$(error NAME is not set. Usage: make create NAME=<project_name>)
endif
# Temporarily replace 'mypkg' with the new project name in all files
	find $(CURDIR)/mypkg -type f -exec sed -i 's/mypkg/$(NAME)/g' {} +
# Temporarily replace 'mypkg' with the new project name in all directories
	find $(CURDIR) -depth -type d -name '*mypkg*' | while read dir; do \
		mv "$$dir" "$$(dirname "$$dir")/$$(basename "$$dir" | sed 's/mypkg/$(NAME)/g')"; \
	done
# Create the zip file
	cd $(CURDIR)/$(NAME) && zip -r ../$(NAME).zip . -x ".git/*" ".vscode/*"
# Revert the directory names
	find $(CURDIR) -depth -type d -name '*$(NAME)*' | while read dir; do \
		mv "$$dir" "$$(dirname "$$dir")/$$(basename "$$dir" | sed 's/$(NAME)/mypkg/g')"; \
	done
# Revert the changes by replacing the new project name back with 'mypkg'
	find $(CURDIR)/mypkg -type f -exec sed -i 's/$(NAME)/mypkg/g' {} +

create-ai: clean-project
ifndef NAME
	$(error NAME is not set. Usage: make create NAME=<project_name>)
endif
# Temporarily replace 'mypkg_ai' with the new project name in all files
	find $(CURDIR)/mypkg_ai -type f -exec sed -i 's/mypkg_ai/$(NAME)/g' {} +
# Temporarily replace 'mypkg_ai' with the new project name in all directories
	find $(CURDIR) -depth -type d -name '*mypkg_ai*' | while read dir; do \
		mv "$$dir" "$$(dirname "$$dir")/$$(basename "$$dir" | sed 's/mypkg_ai/$(NAME)/g')"; \
	done
# Create the zip file
	cd $(CURDIR)/$(NAME) && zip -r ../$(NAME).zip . -x ".git/*" ".vscode/*"
# Revert the directory names
	find $(CURDIR) -depth -type d -name '*$(NAME)*' | while read dir; do \
		mv "$$dir" "$$(dirname "$$dir")/$$(basename "$$dir" | sed 's/$(NAME)/mypkg_ai/g')"; \
	done
# Revert the changes by replacing the new project name back with 'mypkg_ai'
	find $(CURDIR)/mypkg_ai -type f -exec sed -i 's/$(NAME)/mypkg_ai/g' {} +
