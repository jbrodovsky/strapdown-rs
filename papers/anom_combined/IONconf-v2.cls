%%
%% IONconf-v2.cls 2023/06/09 version 2
%%
%% LaTeX class file to generate conference papers for the Institute of
%% Navigation (ION)
%%
%% This class file tries to represent the paper formatting guidelines for ION
%% conference submissions. For latest guidelines please visit www.ion.org.
%%
%% A good first introduction to LaTeX can be found at
%% https://www.overleaf.com/learn/latex/Free_online_introduction_to_LaTeX_(part_1)
%% Further comprehensive documentation on LaTeX can be found at
%% https://ctan.org
%%
%% Compatibility notes
%% This class file requires LaTeX2e.
%% It has been tested with pdfLaTeX and LuaLaTeX compiler engines.
%%
%%*************************************************************************
%% Legal Notice:
%% This code is offered as-is without any warranty either expressed or
%% implied; without even the implied warranty of MERCHANTABILITY or
%% FITNESS FOR A PARTICULAR PURPOSE!
%% User assumes all risk.
%% In no event shall the ION or any contributor to this code be liable for
%% any damages or losses, including, but not limited to, incidental,
%% consequential, or any other damages, resulting from the use or misuse
%% of any information contained here.
%%
%% All comments are the opinions of their respective authors and are not
%% necessarily endorsed by the ION.
%%
%% This work is distributed under the LaTeX Project Public License (LPPL)
%% ( http://www.latex-project.org/ ) version 1.3, and may be freely used,
%% distributed and modified. A copy of the LPPL, version 1.3, is included
%% in the base LaTeX documentation of all distributions of LaTeX released
%% 2003/12/01 or later.
%% Retain all contribution notices and credits.
%% ** Modified files should be clearly indicated as such, including  **
%% ** renaming them and changing author support contact information. **
%%
%% File list of work: IONconf-v2.cls, IONconf_template_userguide.pdf,
%%                    IONconf_template.tex
%%
%%*************************************************************************
%%
%% Available class options (choose one from each category)
% for example \documentclass[letterpaper]{IONconf-v2}
%
%
% letterpaper, a4paper
%   determines the paper size and margins.
%   letterpaper uses US letter size paper 8.5" x 11" with margins:
%   1" top and bottom, 0.75" left and right
%   a4paper uses A4 size paper 210mm x 297mm with margins:
%   1.18" (30mm) top and bottom, 0.59" (15mm) left and right.
%   Paper sizes are defined using the geometry package.
%   Default is letterpaper.
%
% Required font
% times, calibri
% Default is times.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Identification
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{IONconf-v2}[2023/06/09 ION conference paper class]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% class option conditionals
\newif\iftimes
\newif\ifcalibri

%% Preliminary declarations

% base class on article class
\LoadClass[twoside]{article}

% to determine the compiler
\RequirePackage{iftex}

% to be able to get rid of page numbers
\RequirePackage{nopageno}

% page layout
\RequirePackage{geometry}
\setlength\parindent{15pt}%
% for bold table identifiers
\RequirePackage{subcaption}

% for paragraph spacing and indenting
\RequirePackage{parskip}
\RequirePackage{amsmath,amssymb}%
\RequirePackage{graphicx}%
\graphicspath{{./figures/}}
\RequirePackage{cuted}%
\RequirePackage{ifpdf}%
\RequirePackage{xurl}
\urlstyle{same}
\RequirePackage[all]{nowidow}
\RequirePackage{calc}%
\RequirePackage{xcolor}%
\RequirePackage{float}%
\RequirePackage{array}%
\RequirePackage{tabularray}
\UseTblrLibrary{booktabs}
\RequirePackage{tikzpagenodes}
\RequirePackage{tikz}
\RequirePackage[figuresright]{rotating}%
\RequirePackage{afterpage}%
\RequirePackage{booktabs}%
\RequirePackage{enumitem}%
\setlist[enumerate]{nosep,itemsep=4pt,labelindent=\parindent, leftmargin=!}
\setlist[itemize]{nosep,itemsep=4pt,labelindent=\parindent, leftmargin=!,label=$\bullet$}
\RequirePackage{soulutf8}
\sodef\spac{}{.2em}{0.8em plus1em}{0.6em plus.3em minus 0.3em}

\RequirePackage[backend=biber,natbib=true,style=apa,]{biblatex}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Options

\DeclareOption{twocolumn}{\OptionNotUsed} % don't allow two column style

% define paper and font sizes
\DeclareOption{letterpaper}{\geometry{letterpaper,marginparwidth=0in,
                                      total={7in,9in},%{5in,9in},
                                      left=0.75in,%1.75in,%
                                      top=1in}}

\DeclareOption{a4paper}{\geometry{a4paper,marginparwidth=0in,
                                  total={180mm,237mm},
                                  left=15mm,%42mm,%
                                  top=30mm}}

% font times new roman vs. calibri
\DeclareOption{times}{\timestrue}
\DeclareOption{calibri}{\calibritrue}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}} % all other options

% default to US letter paper
\ExecuteOptions{letterpaper,times}
% override defaults by user input
\ProcessOptions

% determine which font to use (times new roman - like or calibri)
\iftimes
%\RequirePackage{mathptmx}
\RequirePackage{newtxtext}
\renewcommand{\normalsize}{\fontsize{10}{12}\selectfont}
\ifcalibri
\RequirePackage[default]{opensans}
\RequirePackage[italic]{mathastext}%
\renewcommand{\normalsize}{\fontsize{10}{14}\selectfont}
\else
\RequirePackage{newtxtext}
\renewcommand{\normalsize}{\fontsize{10}{12}\selectfont}
\fi

%color
\definecolor{tenblack}{cmyk}{0,0,0,0.10}

\def\jmkLabel#1{\@bsphack\protected@write\@auxout{}{\string\Newlabel{#1}{\@currentlabel}}\@esphack}%
\def\Newlabel#1#2{\expandafter\xdef\csname X@#1\endcsname{#2}}%
\def\jmkRef#1{\@ifundefined{X@#1}{0}{\csname X@#1\endcsname}}%

\def\@doi{}%
\def\doi#1{\gdef\@doi{#1}}%

\newcount\emailcount%
\newif\if@auemail%
\newcommand{\email}[1]{Email: #1}%

\newskip\abs@coli@hsize\abs@coli@hsize12pc%0.33\textwidth%
\newskip\abs@colii@hsize\abs@colii@hsize29pc%0.63\textwidth%

%fonts definition
\def\arttypefont{\fontsize{9}{9}\fontseries{b}\selectfont}%
\def\addressfont{\hsize\abs@coli@hsize\fontsize{8}{11}\selectfont\leftskip\z@\rightskip\z@ plus1fil}%
\def\corresfont{\hsize\abs@coli@hsize\fontsize{8}{11}\selectfont\leftskip\z@\rightskip\z@ plus1fil}%
\def\keywordsheadfont{\hsize\abs@colii@hsize\fontsize{10}{10}\selectfont\ifAbstractexist\leftskip7\p@\rightskip\leftskip\fi}%
\def\abstractfont{\hsize\abs@colii@hsize\fontsize{8}{13}\selectfont\leftskip7\p@\rightskip\leftskip}%
\def\absheadfont{\hsize\abs@colii@hsize\fontsize{10}{10}\fontseries{b}\selectfont\bfseries\leftskip7\p@\rightskip\leftskip}%
\def\keywordsfont{\fontsize{8}{13}\selectfont\ifAbstractexist\leftskip7\p@\rightskip\leftskip\fi}%
\def\Authorfont{\fontsize{12}{18}\selectfont\bfseries\leftskip\z@\rightskip\z@ plus1fil}%
\def\titlefont{\fontsize{18}{23}\bfseries\selectfont\leftskip\z@\rightskip\z@ plus1fil\let\mathbcal\titmathbcal}%
\def\titlepageheadfont{\fontsize{7}{9}\selectfont}
\def\runningheadfont{\rmfamily\fontsize{8}{10}\selectfont}
\def\runningfootfont{\fontsize{7}{9}\selectfont}%
\def\sectionfont{\fontsize{12}{13}\fontseries{b}\selectfont\leftskip\z@\rightskip\z@ plus1fil}%
\def\subsectionfont{\fontsize{12}{13}\bfseries\selectfont\leftskip\z@\rightskip\z@ plus1fil}%
\def\subsubsectionfont{\fontsize{12}{13}\bfseries\selectfont\leftskip\z@\rightskip\z@ plus1fil}%
\def\paragraphfont{\fontsize{10.5}{13}\fontseries{b}\selectfont}%
\def\subparagraphfont{\fontsize{10}{13}\fontseries{b}\selectfont}%

\def\artauthors{}

\let\sep\@empty%
\let\authorsep\@empty%
\let\emailsep\@empty%

\newcount\aucount%
\newcount\punctcount%

\def\author{\advance\aucount\@ne%
\@ifstar\@@corrauthor\@@author}%

\newcommand{\@@corrauthor}[2][]{\g@addto@macro\artauthors{\undefined%
\renewcommand{\email}[1]{}%
\hsize=\textwidth\raggedright%
\global\advance\aucount\@ne%%
\global\advance\corrauthcount\@ne%%
\def\@currentlabel{#2}
\jmkLabel{corrauth\thecorrauthcount}%
\Authorfont%
\def\baselinestretch{1}%
\authorsep{#2\ifnum\aucount=2\else\ifnum\punctcount=2,\else\ifnum\punctcount=1\relax\else\ifnum\punctcount=0\relax\else,\fi\fi\fi\fi}%
\unskip\ifx#1\empty\else\textsuperscript{\smash{{%
\@for\@@affmark:=#1\do{\edef\affnum{\@ifundefined{X@\@@affmark}{1}{\jmkRef{\@@affmark}}}%
\unskip\sep\affnum\let\sep=,}}}}\fi%
\def\authorsep{\au@and~}%
\global\let\sep\@empty%
}}%

\newcommand\fnm[1]{{\StepUpCounter{\fnmCnt}#1\addtocontents{aq}{\protect{\bfseries Author\ \thefnmCnt}\hfill\protect\break Given Names: #1\hfill\protect\break}}}%
\newcommand\sur[1]{{#1\addtocontents{aq}{Last Name: #1\protect\par}}}%

\newbox\aufootnotemark%

\newbox\authmarkbox%
\newif\ifGApresent%

\def\@filtername#1#2\@nil{#1.\ }%
\def\filtername#1 #2\@nil{%
\@filtername#1\@nil #2}%
\newcommand{\@@author}[2][]{%
\def\temp{#1}\g@addto@macro\artauthors{%
\def\footnotemark##1{%
\affilmarkdefns%
\global\setbox\authmarkbox\hbox{\reset@font\fontfamily{\rmdefault}\fontsize{8.75pt}{0}\selectfont{##1}}}%
\ifnum\aucount=1%
    \global\@auemailtrue%
    \else%
    \global\@auemailfalse%
    \fi%
    \Authorfont%\
    \def\baselinestretch{1}%
   \authorsep{#2}\unskip\ifx\temp\@empty%
      \@ifundefined{Xname@0}%
        {\expandafter\xdef\csname Xname@0\endcsname{\filtername#2\@nil}}%
        {\expandafter\protected@xdef\csname Xname@0\endcsname%
             {\csname Xname@0\endcsname,\ \filtername#2\@nil}}%
   \else%
   \textsuperscript{\smash{{%
      \@for\@@affmark:=#1\do{\edef\affnum{\@ifundefined{X@\@@affmark}{1}{\jmkRef{\@@affmark}}}%
      \@ifundefined{Xname@\@@affmark}%
        {\expandafter\xdef\csname Xname@\@@affmark\endcsname{\filtername#2\@nil}}%
        {\expandafter\protected@xdef\csname Xname@\@@affmark\endcsname%
             {\csname Xname@\@@affmark\endcsname,\ \filtername#2\@nil}}%
     \unskip\sep\ifnum\addcount=1\else\affnum\fi\let\sep=,}}}\unhbox\authmarkbox}\fi%
    \gdef\authorsep{\enspace$|$\enspace}%%%
    \global\let\sep\@empty\global\let\@corref\@empty%
}}%


\newcounter{affn}%
\renewcommand\theaffn{\arabic{affn}}%

\def\jmkaddress{}%

\gdef\@corresinfo{}%
\def\corres#1{\gdef\@corresinfo{#1}}%

\newcommand{\corresinfo}{{\corresfont\baselineskip11pt\textbf{Correspondence}\par\removelastskip\nointerlineskip\vskip1pt \@corresinfo \par}}%


%% Keywords %%
\def\keywords#1{\def\@keywords{{\keywordsheadfont\textbf{Keywords}\par%
\removelastskip\nointerlineskip\vskip2pt \keywordsfont#1\par}}}\def\@keywords{}%

%
\def\journalname#1{\gdef\@journalname{#1}}\gdef\@journalname{}%
\def\journalvolume#1{\gdef\@journalvolume{#1}}\gdef\@journalvolume{}%
\def\journalnumber#1{\gdef\@journalnumber{#1}}\gdef\@journalnumber{}%
\def\received#1{\gdef\@received{Received: #1}}\gdef\@received{}%
\def\revised#1{\gdef\@revised{Revised: #1}}\gdef\@revised{}%
\def\accepted#1{\gdef\@accepted{Accepted: #1}}\gdef\@accepted{}%

\newif\ifarticletype\global\articletypefalse

\def\pagenumberalign#1{\hbox to 22.5\p@{#1}}%
\def\rhvertbar{\smash{\raisebox{-2\p@}{\rule{.5\p@}{10\p@}}}}%
\def\historydatesvertbar{\hspace*{7\p@}\smash{\raisebox{-4\p@}{\rule{.5\p@}{12\p@}}}\hspace*{7\p@}}%
\def\foot@pagenumber@align#1{\hbox to 15.75\p@{#1}}%
\def\foot@vertbar{\hspace*{7\p@}\smash{\raisebox{0\p@}{\rule{.5\p@}{9.25\p@}}}\hspace*{7\p@}}%
\def\authorseprule#1{\gdef\author@seprule{\hspace*{7\p@}#1\hspace*{7\p@}}}\authorseprule{\smash{\raisebox{.1\p@}{\hbox{\rule{.5\p@}{8.5\p@}}}}}%
%
\def\thepagedef{{\pagenumfont\thepage}}%

\def\headfootrule#1{\gdef\@headfootrule{#1}}\headfootrule{{\rule{\textwidth}{.5\p@}}}%

\def\historykern#1{\gdef\history@kern{#1}}\historykern{\nobreakspace{}\kern.5pt}
\def\@dummy@received{\textcolor{red}{Received: Missing}}%
\def\@dummy@revised{\textcolor{red}{Revised: Missing}}%
\def\@dummy@accepted{\textcolor{red}{Accepted: Missing}}%
\def\@dummy@doi{\textcolor{blue}{DOI: Missing}}%
\def\@dummy@journalname{\textcolor{red}{Journal name: Missing}}%
\def\@dummy@journalvolume{\textcolor{red}{Volume: Missing}}%
\def\@dummy@journalnumber{\textcolor{red}{Number: Missing}}%
\def\@dummy@articletype{\textcolor{red}{Article type: Missing}}%
\def\historydates#1{\gdef\@history@dates{#1}}%
\historydates{{\titlepageheadfont%
  \ifx\@received\@empty\@dummy@received\else\@received\fi\historydatesvertbar%
  \ifx\@revised\@empty\@dummy@revised\else\@revised\fi\historydatesvertbar%
  \ifx\@accepted\@empty\@dummy@accepted\else\@accepted\fi%\historydatesvertbar%
}}%
%
\def\doiheadtext#1{\gdef\@DOI@text{#1}}\doiheadtext{DOI:\nobreakspace{}}%xxx/xxxx
\def\head@titlepage@def#1#2{\vbox{%
  \hbox to \textwidth{#1\dummyalphabet}\par\vspace*{2.5\p@}%
  \@headfootrule\par\vspace*{9\p@}%
  \smash{\ifx\@doi\@empty\titlepageheadfont\@dummy@doi\else{\titlepageheadfont\@DOI@text#2}\fi}\par\vspace*{9\p@}%
  \vspace*{-2.5\p@}%
}}%
%
\newskip\foot@titlepage@raise@skip\global\foot@titlepage@raise@skip23\p@%

\def\oddhead@titlepage@info{\head@titlepage@def{\@history@dates\hfill}{\@doi}}%
\def\evenhead@titlepage@info{\oddhead@titlepage@info}%
%
\def\foot@titlepage@info{%
\begin{tikzpicture}[remember picture,overlay,font=\runningfootfont]
\draw [black](current page footer area.north west)--
(current page footer area.north east);
\node[anchor=west,text width=0.4\textwidth,inner xsep=0pt]
at (current page footer area.south west){%
\ifx\@journalname\@empty\@dummy@journalname\,\else\@journalname,~\fi
\ifx\@journalvolume\@empty\@dummy@journalvolume\,\else\@journalvolume\fi
\ifx\@journalnumber\@empty\@dummy@journalnumber\else(\@journalnumber)\fi%
};
\node[anchor=east,inner xsep=0pt]
at(current page footer area.south east) {© 2023 Institute of Navigation};
\node[inner xsep=0pt]
at(current page footer area.south) {Licensed under CC-BY 4.0};
%%%logo
\node[anchor=north east]at(current page text area.north east)
{\includegraphics[scale=1.0]{ion_logo}};
\end{tikzpicture}}

%\def\evenfoot@titlepage@info{\oddfoot@titlepage@info}%

\def\ps@titlepage{%%
  \def\@oddfoot{\foot@titlepage@info}%
  \def\@evenfoot{\foot@titlepage@info}%
  \def\@oddhead{\oddhead@titlepage@info}%
  \def\@evenhead{\evenhead@titlepage@info}%
  \let\@mkboth\markboth%
  %\afterpage{\globaldefs=1 \restoregeometry}
}%

\def\evenhead@ion@info{%
\begin{tikzpicture}[remember picture,overlay,font=\runningheadfont]
\draw [black](current page header area.south west)--
(current page header area.south east);
\node[anchor=east,inner xsep=0pt]
at(current page header area.east) {\textsc{\@authormark}};
%%%logo
\node[anchor=north west,inner sep=0pt,fill=white,yshift=-1.75pt]at(current page header area.north west)
{\includegraphics[scale=1.0]{ion_logo}};
\end{tikzpicture}}

\def\oddhead@ion@info{\begin{tikzpicture}[remember picture,overlay,font=\runningheadfont]
\draw [black](current page header area.south west)--
(current page header area.south east);
\node[anchor=west,inner xsep=0pt]
at(current page header area.west) {\textsc{\@authormark}};
%%%logo
\node[anchor=north east,inner sep=0pt,fill=white,outer sep=0pt,yshift=-1.75pt]at(current page header area.north east)
{\includegraphics[scale=1.0]{ion_logo}};
\end{tikzpicture}}%

\def\ps@ion{%
  \def\@oddfoot{}%
  \def\@evenfoot{}%
  \def\@oddhead{\oddhead@ion@info}%
  \def\@evenhead{\evenhead@ion@info}%
  \let\@mkboth\markboth%
}%

\def\@authormark{}%
\def\authormark#1{%
\gdef\@authormark{\def\etal{\textit{ET AL.}}\let\AQ\AHAQ#1}%
\gdef\@authormarkforcite{\undefined\def\etal{\textit{et al.}}#1}}%
\def\etal{\textit{et al.}}%

% page style
\pagestyle{ion}

\newcount\corrauthcount%
\newcount\addcount% To check the count of address
%
\def\address{\advance\addcount\@ne\@ifstar\@@corraddress\@@address}%
%
\newcommand{\@@corraddress}[2][]{\gdef\@corresinfo{#2}}%
%
\newcommand{\@@address}[2][]{\def\temp{#1}%
\g@addto@macro\jmkaddress{\hsize=\textwidth%
\def\baselinestretch{1}%
\stepcounter{affn}%
\xdef\@currentlabel{\theaffn}%
\jmkLabel{#1}%
\addressfont\ifx#1\empty\else\@hangfrom{\textsuperscript{\smash{\theaffn}}}\fi%
#2\vphantom{Thy}\par%
}}%

%% paper category %%
\newcommand\dunderline[3][-4pt]{{%
  \sbox0{#3}%
  \ooalign{\copy0\cr\rule[\dimexpr#1-#2\relax]{\wd0}{#2}}}}

\def\@DocHead{}%
\def\articletype#1{\gdef\@DocHead{\hsize=\textwidth\arttypefont\bfseries\uppercase{\dunderline{1pt}{\spac{#1}}}}}%

%% article title %%
\def\titlemark#1{\gdef\@titlemark{#1}}%
\gdef\@titlemark{}%

\def\titlefootnotemark#1{\textsuperscript{$\@fnsymbol{#1}$}}%

\def\title#1{\gdef\@title{\hsize=\textwidth%
\titlefont#1\addcontentsline{toc}{chapter}{#1}% for bookmarks
}}%
\def\shrttitle#1{\gdef\@shrttitle{#1}}\shrttitle{}%

\newcount\authfncnt%
\authfncnt=0%
\newcount\authfnTmpCnt%
\authfnTmpCnt=1%

\def\authfndefns{%
\loop\ifnum\authfncnt>0%
   \csname\romannumeral\authfnTmpCnt StoreAuthFnTxt\endcsname\par%
   \StepDownCounter{\authfncnt}%
   \StepUpCounter{\authfnTmpCnt}%
\repeat}%

\renewcommand{\maketitle}{\par%
  \@afterindentfalse%
  \begingroup
      \renewcommand{\thefootnote}{\fnsymbol{footnote}}%
    \def\@makefnmark{\hbox to\z@{$\m@th^{\@thefnmark}$\hss}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hbox to1.8em{\hss$\m@th^{\@thefnmark}$}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
         \twocolumn[\@maketitle]%
      \else
         \twocolumn[\@maketitle]%
      \fi
    \else
      \newpage
      \global\@topnum\z@
      \@maketitle
    \fi
    \thispagestyle{titlepage}\@thanks%
  \endgroup
%
  \setcounter{footnote}{0}%
  \let\thanks\relax
  \let\maketitle\relax\let\@maketitle\relax
  \gdef\@thanks{}\gdef\@author{}\gdef\@title{}
  \@afterheading%
}

\newskip\adjtitleskip%
\adjtitleskip=10pt plus4pt minus2pt
\def\zz#1{\vphantom{#1}}%

% define title, section, subsection headings
\renewcommand{\@maketitle}{{\unskip%
\hsize=\textwidth\parindent=0pt%
\thispagestyle{empty}%
\removelastskip\vspace*{5pt}%
\ifx\@DocHead\empty%
\@dummy@articletype
\else%
  {\@DocHead\par}%
\fi%
\removelastskip\vskip4pt%
\ifx\@title\empty\else{\@title\par}\fi%
\removelastskip\vskip18pt%
\ifnum\aucount>0%
   \global\punctcount\aucount%
   {\artauthors\par}%
   \removelastskip\vskip8.25pt%
%
\fi%
   \removelastskip\vskip19pt%
%
   \printabstract
%%
   \removelastskip\vskip\adjtitleskip%
}}%

\newdimen\FreePageSpace%To find out the FreeSpace
\newdimen\ovsize%

\newskip\adjustskip%
\adjustskip=0pt plus10pt minus1pt%
\newskip\titleskip%

\gdef\ComputeFreeSpaceOnPage{%
\par%
\ifdim\pagetotal=0pt%
   \global\FreePageSpace=\vsize%
   \global\advance\FreePageSpace by -\adjustskip%
\else%
   \FreePageSpace=\pagegoal%
  \ifvoid\footins%
     \global\advance\FreePageSpace by -\pagetotal%
     \global\advance\FreePageSpace by -\adjustskip%
  \else%
    \setbox3=\vbox{\unvbox\footins}%
    \htftbox=\ht3%
    \advance\htftbox by \dp3%
    \global\advance\FreePageSpace by -\pagetotal%
    \global\advance\FreePageSpace by -\htftbox%
    \global\advance\FreePageSpace by -\adjustskip%
  \fi%
\fi%
}%

%%%%%%%%%%%%%%%%%%%%%
%abstract
%%%%%%%%%%%%%%%%%
%
\newbox\abs@boxa%
\newbox\abs@boxb%
\newbox\abs@boxc%
%
\newdimen\abs@dima%
\newdimen\abs@dimb%
\newdimen\abs@dimc%
%
\def\absCOvrule{\vrule width0.4pt}%
\def\authCOvrule{{\color{tenblack}\vrule width\abs@colii@hsize}}%

\def\FWauthCOvrule{{\color{tenblack}\vrule width\textwidth}}%
\def\dummyalphabet{\vphantom{abcdefghijklmnopqrstuvwxyz0123456789}}
%
\long\def\absworkarea#1#2{%
  \setbox\abs@boxa\vbox{\hsize\abs@coli@hsize#1\dummyalphabet}%
  \abs@dima=\ht\abs@boxa%
  \advance\abs@dima by \dp\abs@boxa%
  \advance\abs@dima by -4.5\p@%
  \setbox\abs@boxb\vbox{\hsize\abs@colii@hsize#2\dummyalphabet}%
  \abs@dimb=\ht\abs@boxb%
  \advance\abs@dimb by \dp\abs@boxb%
  \ComputeFreeSpaceOnPage%
  \stripsep\z@%
  \ifdim\abs@dimb>\FreePageSpace%
    \setbox8=\vbox{\unvbox\abs@boxb}%
    \advance\FreePageSpace by -\@firstpage@foot@height%
    %\advance\FreePageSpace by \baselineskip%
    \setbox1=\vsplit8 to \FreePageSpace%
    \setbox1=\vbox{\unvbox1}%
    \setbox2=\vbox{\unvbox8}%
    \ifdim\abs@dima>\FreePageSpace%
      \setbox7=\vbox{\unvbox\abs@boxa}%
      \setbox3=\vsplit7 to \FreePageSpace%
      \setbox3=\vbox{\unvbox3}%
      \setbox4=\vbox{\unvbox7}%
    \fi%
    \gdef\abs@dimchk{\ifdim\ht3>\ht1\ht3\else\ht1\fi}%
    \parbox{\textwidth}{%
      \begin{minipage}[t]{\abs@coli@hsize}%
      \leavevmode\vbox to \abs@dimchk{\ifdim\abs@dima>\FreePageSpace\copy3\else\unvbox\abs@boxa\fi\vfill}%
      \end{minipage}\hfill%
      \hspace*{7\p@}\absCOvrule\hspace*{7\p@}\authCOvrule%
      \begin{minipage}[t]{\abs@colii@hsize}%
      \leavevmode\hspace*{-\textwidth}\vbox to \abs@dimchk{\unvbox1\vfill}%
      \end{minipage}%
      \vspace*{\@firstpage@foot@height}%
    }%
    \clearpage%
    \loop%
    \ifdim\ht2>\textheight%
    \ovsize=\vsize%
    \setbox1=\vsplit2 to \ovsize%
    \setbox1=\vbox{\unvbox1}%
    \setbox3=\vsplit4 to \ovsize%
    \setbox3=\vbox{\unvbox3}%
     \parbox{\textwidth}{%
      \begin{minipage}[t]{\abs@coli@hsize}%
      \leavevmode\vbox{\unvbox3\vfill}%
      \end{minipage}\hfill%
      \hspace*{7\p@}\absCOvrule\hspace*{7\p@}\authCOvrule%
      \begin{minipage}[t]{\abs@colii@hsize}%
      \leavevmode\hspace*{-\textwidth}\vbox{\vspace*{7\p@}\unvbox1\vspace*{7\p@}}%
      \end{minipage}%
   }%
    \clearpage%
    \repeat%
    \gdef\abs@dimchk{\ifdim\ht4>\ht2\ht4\else\ht2\fi}%
     \parbox{\textwidth}{%
      \begin{minipage}[t]{\abs@coli@hsize}%
      \leavevmode\vbox to \abs@dimchk{\copy4\vfill}%
      \end{minipage}
      \hfill%
      \hspace*{7\p@}\absCOvrule\hspace*{7\p@}\authCOvrule%
      \begin{minipage}[t]{\abs@colii@hsize}%
      \leavevmode\hspace*{-\textwidth}\vbox to \abs@dimchk{\unvbox2\vfill}%
      \end{minipage}%
      \vspace*{2pc}%
      \vskip\titleskip%
    }%
  \else%
    \ifdim\abs@dima>\FreePageSpace%
      \setbox8=\vbox{\unvbox\abs@boxa}%
      \advance\FreePageSpace by -\@firstpage@foot@height%
      %
      \setbox1=\vsplit8 to \FreePageSpace%
      \setbox1=\vbox{\unvbox1}%
      \setbox2=\vbox{\unvbox8}%
       \parbox{\textwidth}{%%
        \begin{minipage}[t]{\abs@coli@hsize}%
        \leavevmode\vbox{\copy1\vfill}%
        \end{minipage}\hfill%
        \hspace*{7\p@}\absCOvrule\hspace*{7\p@}\authCOvrule%
        \begin{minipage}[t]{\abs@colii@hsize}%
        \leavevmode\hspace*{-\textwidth}\vbox to \ht1{\unvbox\abs@boxb\vfill}%
        \end{minipage}%
        \vspace*{\@firstpage@foot@height}%
      }%
      \clearpage%
      \loop\ifdim\ht2>\textheight%
      \ovsize=\vsize%
      \setbox1=\vsplit2 to \ovsize%
      \setbox1=\vbox{\unvbox1}%
       \parbox{\textwidth}{%%
        \begin{minipage}[t]{\abs@coli@hsize}%
        \leavevmode\vbox{\unvbox1\vfill}%
        \end{minipage}\hfill%
        \hspace*{7\p@}\absCOvrule\hspace*{7\p@}\authCOvrule%
        \begin{minipage}[t]{\abs@colii@hsize}%
        \leavevmode\hspace*{-\textwidth}\vbox{\vspace*{7\p@}\unvbox1\vspace*{7\p@}}%
        \end{minipage}%
      }%
      \clearpage%
      \repeat%
       \parbox{\textwidth}{%
        \begin{minipage}[t]{\abs@coli@hsize}%
        \leavevmode\vbox{\unvbox2\vfill}%
        \end{minipage}
        \hfill%
        \hspace*{7\p@}\absCOvrule\hspace*{7\p@}\authCOvrule%
        \begin{minipage}[t]{\abs@colii@hsize}%
        \leavevmode\hspace*{-\textwidth}\vbox{\vspace*{7\p@}\unvbox2\vspace*{-7\p@}}%
        \end{minipage}%
        \vspace*{2pc}%
        \vskip\titleskip%
      }%
    \else%
      \gdef\abs@dimchk{\ifdim\abs@dima>\abs@dimb\abs@dima\else\abs@dimb\fi}%
       \parbox{\textwidth}{%%
        \begin{minipage}[t]{\abs@coli@hsize}%
        \leavevmode\vbox to \abs@dimchk{\copy\abs@boxa\vfill}%
        \end{minipage}\hfill%
        \hspace*{7\p@}\absCOvrule\hspace*{7\p@}\authCOvrule%
        \begin{minipage}[t]{\abs@colii@hsize}%
        \leavevmode\hspace*{-\textwidth}\vbox to \abs@dimchk{\unvbox\abs@boxb\vfill}%
        \end{minipage}%
        \vspace*{2pc}%
        \vskip\titleskip%
      }%
    \fi%
  \fi%
%%%
}%

%\renewcommand{\abstractname}{Abstract}

\newif\ifAbstractexist%
\global\Abstractexistfalse%

%% Abstract
\renewcommand\abstract[2][Abstract]{\global\Abstractexisttrue%
  \gdef\@abstract{%
    \setlength{\parindent}{\z@}%
    \vspace*{6.5\p@}%
    \rightskip0pt\raggedright\hspace*{7\p@}\hbox{\reset@font\abstractfont{\absheadfont#1}}\par\vskip-4pt% below Abstract name
    {\abstractfont\baselineskip11pt\ifFWabstract\hsize\textwidth\fi#2\par\vspace*{0\p@}}%
    \addcontentsline{toc}{section}{\abstractname}%
}}%

\newif\ifabstract\global\abstractfalse%
\newif\ifFWabstract%
%
\gdef\printabstractpart{%
  \absworkarea{%
    \noindent{\reset@font\addressfont\jmkaddress\par}%
 \removelastskip\vskip8pt%
   \ifx\@corresinfo\empty\else{\corresinfo\par}\fi%
   \removelastskip\vskip8pt%
  }{%
    \global\abstracttrue%
    \ifx\@abstract\@empty\else\@abstract\fi%
\removelastskip\vskip10pt%
    \ifx\@keywords\@empty\else\@keywords\par\vskip-6pt%\fi%
    \global\abstractfalse%
  }
}

%% Printing Abstract
\long\def\printabstract{%
  \bgroup%
    \setlength{\parindent}{\z@}%
    \printabstractpart%
  \egroup%
}

% \renewcommand{\section}{
%   \@startsection{name}
%   {level}
%   {indent}
%   {beforeskip}
%   {afterskip}
%   {style}}

\renewcommand\section{\@startsection{section}{1}{\z@}{-27pt \@plus -2pt \@minus -2pt}{12\p@}{\sectionfont}}%
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}{-23pt \@plus -2pt \@minus -2pt}{5\p@}{\subsectionfont}}%
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}{-20pt \@plus -2pt \@minus -2pt}{2\p@}{\subsubsectionfont}}%
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}{-10pt \@plus -2pt \@minus -2pt}{2\p@}{\paragraphfont}}%
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\z@}{-10pt \@plus -2pt \@minus -2pt}{-12\p@}{\subparagraphfont}}%

% define section numbers in headlines
\renewcommand\thesection      {\@arabic\c@section}
\renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection.\@arabic\c@subsubsection}
\renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}
\renewcommand\thesubparagraph {\theparagraph\@arabic\c@subparagraph}


\newskip\secruleskip\secruleskip8.5\p@%
\newif\ifappendixsec%

\def\secrule{\hspace*{\secruleskip}\raisebox{0pt}{\rule{.5\p@}{7.5\p@}}\hspace*{\secruleskip}}%
\def\subsecrule{\secrule}%
\def\subsubsecrule{\hspace*{\secruleskip}\raisebox{0pt}{\rule{.5\p@}{6.5\p@}}\hspace*{\secruleskip}}%


\def\@sectioncntformat#1{{\ifappendixsec\csname the#1\endcsname\ \else\csname the#1\endcsname\secrule\fi}}%
\def\@subsectioncntformat#1{{\ifappendixsec\csname the#1\endcsname\ \else\csname the#1\endcsname\subsecrule\fi}}%
\def\@subsubsectioncntformat#1{{\ifappendixsec\csname the#1\endcsname\ \else\csname the#1\endcsname\subsubsecrule\fi}}%

\def\@paragraphcntformat#1{{\ifappendixsec\csname the#1\endcsname\ \else\csname the#1\endcsname.\ \fi}}%
\def\@subparagraphcntformat#1{#1}

\def\@sect#1#2#3#4#5#6[#7]#8{%
   \ifnum #2>\c@secnumdepth%
      \edef\@svsec{\noindent}%
   \else%
      \refstepcounter{#1}%
      \let\@@protect\protect%
      \def\protect{\noexpand\protect\noexpand}%
      \edef\@svsec{\csname @#1cntformat\endcsname{#1}}%
      \let\protect\@@protect%
   \fi%
   \@tempskipa #5\relax%
   \ifdim \@tempskipa>\z@%
      \ifnum#2=1%
         \begingroup #6\relax%\@hangfrom%
            {\noindent\hskip #3\relax{\@svsec}}%
            {\interlinepenalty\@M{\uppercase{#8}}\par}%
         \endgroup%
      \else%
         \ifnum#2=2%
            \begingroup #6\relax%\@hangfrom%
               {\noindent\hskip #3\relax\@svsec}%
               {\interlinepenalty\@M{#8}\par}%
            \endgroup%
         \else%
      \ifnum#2=3%
            \begingroup #6\relax%\@hangfrom%
               {\noindent\hskip #3\relax\@svsec}%
               {\interlinepenalty\@M{#8}\par}%
            \endgroup%
         \else%
      \ifnum#2=4%
            \begingroup #6\relax%\@hangfrom%
               {\noindent\hskip #3\relax\@svsec}%
               {\interlinepenalty\@M{\textcolor{black!75}{#8}}\par}%
            \endgroup%
         \else%
            \begingroup #6\relax%\@hangfrom
               {\noindent\hskip #3\relax\@svsec}%
               {\interlinepenalty\@M#8\par}%
            \endgroup%
         \fi%
     \fi%
     \fi%
      \fi%
      \csname #1mark\endcsname{#7}%
      \addcontentsline{toc}{#1}{%
        \ifnum #2>\c@secnumdepth \else%
          \protect\numberline{\csname the#1\endcsname}%
        \fi%
        #7}%
   \else%
      \def\@svsechd{%
         #6{\hskip #3\relax%
         \@svsec \textcolor{black!75}{#8}\hspace*{4pt}}%
      }%
   \fi%
   \@xsect{#5}%
}

\RequirePackage{microtype}
\SetTracking[no ligatures={f},spacing={1100*,-100*,},outer spacing={450,250,150},outer kerning={*,*}]{encoding=*}{200}

\def\@ssect#1#2#3#4#5{\@tempskipa #3\relax
  \ifdim \@tempskipa>\z@
     \begingroup \noindent #4{\relax{\hskip #1}{\interlinepenalty\@M{\uppercase{\textls{#5}}\@@par}}}%
     \endgroup
  \else \def\@svsechd{#4{\hskip #1\relax #5 \mbox{}}}\fi
  \@xsect{#3}}

%Figure caption
\sodef\spacfig{}{.15em}{.8em plus1em}{0.6em plus.3em minus 0.3em}
\DeclareCaptionLabelFormat{mylabel}{#1 #2\hspace{1.25ex}}

\captionsetup[figure]{labelformat=mylabel, labelsep=none, skip=10pt, labelfont={bf,small,color=black!70},
justification=raggedright,singlelinecheck=off,font=small,}
\captionsetup[table]{labelformat=mylabel, labelsep=newline, skip=10pt, labelfont={bf,small,color=black!70},
justification=raggedright,singlelinecheck=off,font=small,}

\renewcommand{\thefigure}{\arabic{figure}}

\renewcommand{\thetable}{\arabic{table}}

\renewcommand\figurename{\textbf{\spacfig{FIGURE\,}}}%%
\renewcommand\tablename{\textbf{\spacfig{TABLE\,}}} %%

\sloppy%
\frenchspacing%

%%% special parameters for TeX
\adjdemerits=100
\linepenalty=100

% Page break penalties
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301

% Disallow widows and orphans
\clubpenalty 10000
\widowpenalty 10000

% Disable page breaks before equations, allow pagebreaks after
% equations and discourage widow lines before equations.
\displaywidowpenalty 100
\predisplaypenalty   10000
\postdisplaypenalty  0

% Set these global demerits
\doublehyphendemerits 1000000% corresponds to badness 800
\finalhyphendemerits  1000000% corresponds to badness 1000

% Allow loose lines rather than overfull lines
\vbadness=9999%
\tolerance=9999%


% Allow breaking the page in the middle of a paragraph
\interlinepenalty 0%

% Disallow breaking the page after a hyphenated line
\brokenpenalty 10000%

% Hyphenation; don't split words into less than three characters
\lefthyphenmin=3%
\righthyphenmin=3%

% Float placement parameters

% The total number of floats that can be allowed on a page.
\setcounter{totalnumber}{10}%

% The maximum number of floats at the top and bottom of a page.
\setcounter{topnumber}{3}%
\setcounter{bottomnumber}{3}%

% The maximum part of the top or bottom of a text page that can be
% occupied by floats. This is set so that at least four lines of text
% fit on the page.
\renewcommand\topfraction{.921}%
\renewcommand\bottomfraction{.921}%

% The minimum amount of a text page that must be occupied by text.
% This should accomodate four lines of text.
\renewcommand\textfraction{.1}%

% The minimum amount of a float page that must be occupied by floats.
\renewcommand\floatpagefraction{.95}%

% The same parameters repeated for double column output
\renewcommand\dbltopfraction{.95}%
\renewcommand\dblfloatpagefraction{.95}%

% Space between floats
\setlength\floatsep{24\p@ \@plus 2.875\p@ \@minus 1.4375\p@}%

% Space between floats and text
\setlength\textfloatsep{24\p@ \@plus 2.875\p@ \@minus 1.4375\p@}%
%
% Space above and below an inline figure
\setlength\intextsep{24\p@ \@plus 2.875\p@ \@minus 1.4375\p@}%
