# Setup 

## Installing stow
Using stow as a symlink manager
`sudo apt-get install stow`

## Syncing
```
cd ~
git clone https://github.com/bkoch4142/dotfiles
cd dotfiles
stow vim  #make sure files you stow don't already exist
```

## Vimspector Setup
- Config is in .vimspector.json
- Make sure the python path matches
```
vim
:VimspectorInstall --enable-python
```

## Other
- if you are on windows terminal disable paste cntrl-v, go to settings and edit settings json, comment out // { "command": "paste", "keys": "ctrl+v" }, <------ THIS LINE
- if colors not showing maybe its bash fault so add: export TERM=xterm-256color

## TODO
- also checkout vim-dirvish (better than netrw)
- check jupyter vim binding
- check tpope plugins
- check fzf
- check undotree
- make undodir config
- lightline
- python syntax
- neovim
- coc.nvim
- https://www.youtube.com/watch?v=gnupOrSEikQ&feature=emb_title&ab_channel=BenAwad
- learn the c command for editing everything found
- image view with xterm and vifm
