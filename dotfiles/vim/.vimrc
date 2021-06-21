" Vimrc
let mapleader = ","

" set line numbers
set nu

" Fast saving
nmap <leader>w :w!<cr>

" Background
set background=dark
" Makes the colorscheme work in tmux
" set t_Co=256

"Do not create backup files
set nobackup

" Window commands
nnoremap <C-h> <C-w>h " Remap window change to Ctrl+movement_key
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l
nnoremap <leader>pv :wincmd v<bar> :Ex <bar> :vertical resize 30<CR> " Opens file explorer with pv
set splitbelow " Makes windows open below and not up

" Terminal
map <leader>tt :vnew terminal<CR>

" Remap arrow keys to resize window
nnoremap <Up>    :resize -2<CR>
nnoremap <Down>  :resize +2<CR>
nnoremap <Left>  :vertical resize -2<CR>
nnoremap <Right> :vertical resize +2<CR>

" Per default, netrw leaves unmodified buffers open. This autocommand
" deletes netrw's buffer once it's hidden (using ':q', for example)
autocmd FileType netrw setl bufhidden=delete

"auto search when typing
set incsearch

" wildmenu for command line completion
set wildmenu " ?

" always show current position
set ruler

" dont create swap files
set noswapfile

"ignore case when searching
set ignorecase
set smartcase " case insensitive search until you put an uppercase letter

" highlight search results
set hlsearch

"show matching brackets when cursor is on them
set showmatch

"no sounds
set noerrorbells
set novisualbell
set t_vb=
set tm=500

" enable syntax highlight
syntax enable " ?

" text, tab, indent
set expandtab "uses spaces instead of tabs
set smarttab "smart use of tabs
set shiftwidth=4 "tab to 4 spaces
set tabstop=4 "tab to 4 spaces

"auto indenting
set ai
set si 
set nowrap "let the line go off the screen dont wrap

"remap 0 to go to first non-black char
map 0 ^

"map jj to esc 
imap jj <Esc>

"map ReplaceAll to S
nnoremap S :%s//g<Left><Left>

" To aid in pasting text from outside vim toggle set paste
" :set paste
" :set nopaste

" https://www.youtube.com/watch?v=XA2WjJbmmoM&t=3513s&ab_channel=thoughtbot
" everything below is from the above link

"when looking for a file search for every subdir
"display all matching files when we tab complete
"now you can search with :find deletethis.txt 
"you can also use fuzzy matching
set path+=** 

" you can list everythihg vim has open with :ls
" jump to some file with substring :b substring


"tweeks for browsing
" now you can:
"   :edit folder_name 
"   <CR>/v/t to open in a h-split/v-split/tab
"   check |netrw-browse-maps| for more mappings
let g:netrw_banner=0
let g:netrw_browse_split=4
let g:netrw_altv=1
let g:netrw_liststyle=3
let g:netrw_list_hide=netrw_gitignore#Hide()
let g:netrw_list_hide.=',\(^|\s\s\)\zs\.\S\+'
let g:netrw_localrmdir='rm -r'


" lightline
set laststatus=2
let g:lightline = {
      \ 'colorscheme': 'seoul256',
      \ }

" For Mac/Linux users
call plug#begin('~/.vim/bundle')
if empty(glob('~/.vim/autoload/plug.vim'))
      silent !curl -fLo ~/.vim/autoload/plug.vim --create-dirs
          \ https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
      autocmd VimEnter * PlugInstall --sync | source $MYVIMRC
endif

Plug 'itchyny/lightline.vim'
Plug 'morhetz/gruvbox'
Plug 'vifm/vifm.vim'
Plug 'puremourning/vimspector'
Plug 'szw/vim-maximizer'

call plug#end()

" vimspector
let g:vimspector_enable_mappings = 'HUMAN'

" Maximizes a vim window 
nnoremap <leader>m :MaximizerToggle!<CR>
" Launches Debug
nnoremap <leader>dd :call vimspector#Launch()<CR>
" Exites Debug
nnoremap <leader>de :call vimspector#Reset()<CR>
nnoremap <leader>dtcb :call vimspector#CleanLineBreakpoint()<CR>
" Flow Control commands
nmap <leader>dl <Plug>VimspectorStepInto
nmap <leader>dj <Plug>VimspectorStepOver
nmap <leader>dk <Plug>VimspectorStepOut
nmap <leader>d_ <Plug>VimspectorRestart
nnoremap <leader>d<space> :call vimspector#Continue()<CR>
nmap <leader>drc <Plug>VimspectorRunToCursor
" Breakpoints
nmap <leader>dbp <Plug>VimspectorToggleBreakpoint
nmap <leader>dcbp <Plug>VimspectorToggleConditionalBreakpoint
" Inspect
nmap <leader>di <Plug>VimspectorBalloonEval
xmap <leader>di <Plug>VimspectorBalloonEval

" Colorcheme
colorscheme gruvbox
