%% 
%% IONconf.cls 2021/06/10 version 1.1
%% 
%% LaTeX class file to generate conference papers for the Institute of 
%% Navigation (ION)
%% 
%% IONconf.cls is written and maintained by Fabian Rothmaier, Stanford University.
%% Please contact fabianr@alumni.stanford.edu with any questions, error reports 
%% or suggestions.
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
%% 
%% 
%% Acknowledgements
%% Much of this template is inspired by Michael Shell's excellent IEEEtran 
%% template.
%% Realization of this work would further not have happened without the 
%% positive encouragement and support of Prof. Todd Walter, Rick Buongiovanni
%% and many more of the ION staff.
%% 
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
%% File list of work: IONconf.cls, IONconf_template_userguide.pdf, 
%%                    IONconf_template.tex
%% 
%%*************************************************************************
%% 
%% Available class options (choose one from each category)
% for example \documentclass[letterpaper]{IONconf}
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
% 
% times, calibri
%   determines the font to be used.
%   ION conference paper guidelines specify that either Times New Roman or 
%   Calibri font should be used.
%   Note:
%   The pdfLaTeX compiler is not compatible with the current implementation
%   of the Calibri font. To use the Calibri font option, use the LuaLaTeX
%   compiler. The Calibri font files further need to be saved in a folder
%   ./fonts/Calibri/. A different folder location can be specified  by 
%%  redefining the \CalibriFontFolderPath command.
%   Required font files are:
%       Calibri.ttf
%       CalibriBold.ttf
%       CalibriBoldItalic.ttf
%       CalibriItalic.ttf
%   Default is times.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Identification
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{IONconf}[2021/06/10 ION conference paper class]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% class option conditionals
\newif\ifCLSOPTcalibri      \CLSOPTcalibrifalse % default to false
\newif\ifCLSOPTpdflatex     \CLSOPTpdflatextrue % default is true

%% Calibri folder path
\newcommand{\CalibriFontFolderPath}{./fonts/Calibri/}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Preliminary declarations

% base class on article class
\LoadClass{article}

% require some packages

% to determine the compiler
\RequirePackage{iftex}

% to be able to get rid of page numbers
\RequirePackage{nopageno}

% page layout
\RequirePackage{geometry}

% for bold table identifiers
\RequirePackage[font=small, labelfont=bf, tableposition=top]{caption}

% for paragraph spacing and indenting
\RequirePackage{parskip}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Options

\DeclareOption{twocolumn}{\OptionNotUsed} % don't allow two column style

% define paper and font sizes
\DeclareOption{letterpaper}{\geometry{letterpaper,
                                      total={7in,9in},
                                      left=0.75in,
                                      top=1in}}

\DeclareOption{a4paper}{\geometry{a4paper,
                                  total={180mm,237mm},
                                  left=15mm,
                                  top=30mm}}

% font times new roman vs. calibri
\DeclareOption{times}{\CLSOPTcalibrifalse}

\DeclareOption{calibri}{\CLSOPTcalibritrue}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}} % all other options

% default to US letter paper
\ExecuteOptions{letterpaper,times}
% override defaults by user input
\ProcessOptions


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% More declarations

% remove page numbers
\pagestyle{plain} 

% check for font folder
\IfFileExists{\CalibriFontFolderPath Calibri.ttf}{}{\CLSOPTcalibrifalse}

% determine which compiler is used
\ifluatex
    \CLSOPTpdflatexfalse
\fi
\ifxetex
    \CLSOPTpdflatexfalse
\fi

% determine which font to use (times new roman - like or calibri)
\ifCLSOPTpdflatex
    % \RequirePackage{mathptmx}
    \RequirePackage{newtxtext}
\else
    \RequirePackage{fontspec}
    % pick calibri or Times
    \ifCLSOPTcalibri
        \setmainfont{Calibri}[
            Path            = \CalibriFontFolderPath,
            Extension       = .ttf,
            BoldFont        = *Bold,
            ItalicFont      = *Italic, 
            BoldItalicFont  = *BoldItalic]
    \else
        \setmainfont{TeX Gyre Termes}
    \fi
\fi

% update spacing after equations
\setlength{\belowdisplayskip}{15pt}


% set normal font size to 10
\renewcommand{\normalsize}{\fontsize{10}{11}\selectfont}


% create biography environment
\newenvironment{biography}[2]{
    \textbf{#1} #2%
}

% define title, section, subsection headings
\renewcommand{\maketitle}{
    \begin{center}
        \fontsize{24}{28}\selectfont%
        \textbf{\@title}\par%
        \normalsize%
        \vskip 3ex%
        \begin{tabular}[t]{c}%
            \@author%
        \end{tabular}\par%
    \end{center}
}


% \renewcommand{\section}{
%   \@startsection{name}
%   {level}
%   {indent}
%   {beforeskip}
%   {afterskip}
%   {style}}

\renewcommand{\section}{%
    \@startsection
    {section}{1}{0pt}{4ex plus -1ex minus -.2ex}%
    {1ex plus .2ex}{\bfseries\aftergroup\@afterindentfalse\MakeUppercase}%
}

\renewcommand{\subsection}{%
    \@startsection
    {subsection}{2}{0pt}{3ex plus -1ex minus -.2ex}%
    {1ex plus .2ex}{\bfseries\aftergroup\@afterindentfalse}%
}

\renewcommand{\subsubsection}{%
    \@startsection
    {subsubsection}{3}{0pt}{2ex plus -1ex minus -.2ex}%
    {1ex plus .2ex}{\it\aftergroup\@afterindentfalse}%
}

% define basic sections
\def\thesection{\Roman{section}}                               % I, II, ...
\def\thesubsection{\arabic{subsection}}                        % 1, ...
\def\thesubsubsection{\alph{subsubsection})}                   % a), ...

% add dot to section numbers in headlines
\renewcommand{\@seccntformat}[1]{%
  \csname the#1\endcsname% Print sectional counter
  \ifnum\pdfstrcmp{#1}{section}=0 .\fi% If \section, print .
  \ifnum\pdfstrcmp{#1}{subsection}=0 .\fi% if \subsection, print .
  \enskip% Space between number and title
}

% repair referencing of subsections
\def\p@subsection{\thesection.}
\def\p@subsubsection{\thesection.\thesubsection\,}

