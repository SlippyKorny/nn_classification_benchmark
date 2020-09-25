\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ijisa}[2020/05/05 Latex class for IJISA journal submission]

%% Article options
\DeclareOption{10pt}{
	\PassOptionsToClass{\CurrentOption}{article}
}

%% Process given options
\ProcessOptions\relax

%% Load base
\LoadClass[twoside, twocolumn, 10pt]{article}


%% maketitle

%%Preliminary definitions, needed by the options


%%Body of the class, most of the declarations appear here.

%% page configuration
\setlength{\hoffset}{-1in}
\setlength{\voffset}{-1in}
\setlength{\oddsidemargin}{2cm}
\setlength{\evensidemargin}{2cm}
\setlength{\topmargin}{0.7cm}
\setlength{\headheight}{2.2cm}
\setlength{\headsep}{0.4cm}
\setlength{\textheight}{21.7cm}
\setlength{\textwidth}{17.5cm}
\setlength{\footskip}{1.0cm}
\setlength{\columnsep}{0.5cm}


%% author 
%\usepackage[left=3.00cm, right=3.00cm, top=2.00cm, bottom=2.00cm]{geometry}
\usepackage[blocks]{authblk}
\setlength{\affilsep}{-10pt}
\renewcommand*{\Authsep}{, }
\renewcommand\Authands{ and }
%\renewcommand*{\Affilfont}{\normalsize\normalfont}
\renewcommand*{\Authfont}{\fontsize{11}{11} \selectfont \bfseries}
\renewcommand{\Affilfont}{\fontsize{10}{10} \selectfont}



%% title 
\renewcommand{\maketitle}{
	
	\twocolumn[%
	\begin{center}%
	\fontsize{24}{24} \selectfont \@title 
	\medskip	
	\vspace{0.6cm} \par
	\begin{tabular}{c}
		\setstretch{0.1}
		\baselineskip=2pt
		\@author
		%%\relaxpager
	\end{tabular}
	\vspace{0.6cm}
	\end{center}
	]%   
	\vspace{0.4cm}
%%	\vspace{0.5cm}
	\noindent {\fontsize{9}{12} \selectfont {\bfseries{Abstract}}---\@abstract} \par
	\vspace{0.3cm}
	\noindent {\fontsize{9}{12} \selectfont {\bfseries{Index Terms}}---\@keywords} \par

}


%% Load other package
\usepackage[T1]{fontenc}
\usepackage{ae,aecompl}
\usepackage{pslatex}
\usepackage[samesize]{cancel}
\usepackage{graphicx}
\usepackage{setspace}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{natbib}
\usepackage{float}
\usepackage{mathptmx} 
\usepackage{url}
\RequirePackage{booktabs}
\usepackage{calc}


%% Definition variable 
\def\thisvolume#1{\gdef\@thevolume{#1}}        % Volumen
%\def\thisnumber#1{\gdef\@thenumber{#1}}         % 
\def\thismonth#1{\gdef\@themonth{#1}}             % 
\def\thisyear#1{\gdef\@theyear{#1}}             % 
%\def\receptiondate#1{\gdef\@receptiondate{#1}}       % 
%\def\acceptancedate#1{\gdef\@acceptancedate{#1}}     % 
%\def\publicationdate#1{\gdef\@publicationdate{#1}}   % 
\def\abstract#1{\gdef\@abstract{#1}}        % 
\def\keywords#1{\gdef\@keywords{#1}}        % 
\def\thisstartpage#1{\gdef\@thestartpage{#1}} 
\def\thisendpage#1{\gdef\@theendpage{#1}} 
\def\thisdoi#1{\gdef\@thedoi{#1}} 

\newcommand{\vcenteredinclude}[1]{\begingroup
	\setbox0=\hbox{\includegraphics[height=\headheight]{#1}}%
	\parbox{\wd0}{\box0}\endgroup}

%% header
\usepackage{fancyhdr, color}



%%% Configuración de encabezado y pie de páginas
\usepackage{fancyhdr}
\pagestyle{fancy}
\renewcommand{\headrulewidth}{0pt}

\fancyhead[LE, LO]{ }
\fancyhead[CE, CO]{{\fontsize{7}{10} \selectfont \MakeUppercase{\@title}}}
\fancyhead[RO,RE]{{\fontsize{7}{10} \selectfont \thepage}}
%\fancyhead[RE]{{\fontsize{7}{10} \selectfont \authorheader}}
% \fancyfoot[LO,LE]{Copyright © \@theyear MECS }
\fancyfoot[CO,CE]{ }
% \fancyfoot[RO,RE]{ \textit{I.J. Intelligent Systems and Applications, \@theyear, \@thevolume, \@thestartpage-\@theendpage}}


\fancypagestyle{thefirstpage}{
	\fancyhead{}
% 	\fancyhead[LO]{{\fontsize{7}{10} \selectfont \textbf{\textit{I.J. Intelligent Systems and Applications}}, \@theyear, \@thevolume, \@thestartpage-\@theendpage}\\
% 		{\fontsize{10}{20} \selectfont Published Somewhere \@themonth \@theyear in Somewhere 2 } \\ Doi:\@thedoi }
	\fancyhead[CO]{ }
% 	\fancyhead[RO]{\vcenteredinclude{ijisalogo.png}}
}




%% section 
\usepackage{indentfirst}
\setcounter{secnumdepth}{2}      % Deshabilitar numeración

 
\renewcommand{\section}{\@startsection{section}{1}{0mm}%
	{-12pt}%
	{6pt}%
	{\setlength\parindent{0em}  \centering\textsc}}
	{\setlength\parskip{1pt}}
\renewcommand{\subsection}{\@startsection {subsection}{2}{0mm}%
	{-12pt}%
	{2pt}%
	{\setlength\parindent{0em}  \textit}}%
	{\setlength\parskip{1pt}}

\renewcommand{\subsubsection}{\@startsection{subsubsection}{0mm}%
	{-12pt}%
	{2pt}%
	{\setlength\parindent{0em}  \selectfont \textit}}

\renewcommand\thesection{\Roman{section}}
\renewcommand\thesubsection{\Alph{subsection}}



%% table and figure

\usepackage{caption}
\captionsetup[figure]{font=small,labelfont=bf,justification=centering}
\captionsetup[figure]{name=Fig.}
%\DeclareCaptionFormat{myformat}{#1#2#3\rule{\linewidth}{.3pt}}
%\captionsetup[figure]{format=myformat}

\captionsetup[table]{font={small,sc},labelfont={bf,sc},justification=centering}
\captionsetup[table]{name=Table}

\usepackage{makecell} % break lines in cell tables
\renewcommand\theadalign{cc}
\renewcommand\theadgape{\Gape[4pt]}
\renewcommand\cellgape{\Gape[4pt]}
\addtolength{\belowcaptionskip}{-8pt}
\addtolength{\abovecaptionskip}{0pt}

%% footnote
\renewcommand*\footnoterule{}
\newcommand\blfootnote[1]{%
	\begingroup
	\renewcommand\thefootnote{}\footnote{#1}%

	\endgroup
}

%biography
\usepackage{wrapfig}%




\endinput