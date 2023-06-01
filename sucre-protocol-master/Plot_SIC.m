function Plot_SIC_results=Plot_SIC(MarkerSize,LineWidth,FontSize,EnergiesdB,DataMatrix,NumofFiles,LineMethod,PlotMethod)
TitleString='';
LegendString='1';
% MarkerSize=10;
SizeAll=FontSize;
xlabelSize=SizeAll;
ylabelSize=SizeAll;
LegendStringSize=SizeAll;
TitleStringSize=SizeAll;
axesFontSize=SizeAll;
xlabelString='SNR (dB)';
ylabelString='BER';
TileString = '';
%solidLine=1;
% LineWidth=2;
Rows=size(DataMatrix,1);
for kk=1:Rows/NumofFiles
       ber_ml = DataMatrix((kk-1)*NumofFiles+1:kk*NumofFiles,:);      
    MF_string=PlotWithString(EnergiesdB,ber_ml,TileString,LegendString,MarkerSize,xlabelSize,ylabelSize,LegendStringSize,TitleStringSize,...,
    axesFontSize,LineWidth,xlabelString,ylabelString,LineMethod,PlotMethod);
end
grid on;
Plot_SIC_results=1;
function MF=PlotWithString(SNR,DataMatrix,TitleString,LegendString,MarkerSize,xlabelSize,ylabelSize,LegendStringSize,TitleStringSize,...,
    axesFontSize,LineWidth,xlabelString,ylabelString,solidLine,SemiLog)
%该函数的主要功能是画图程序
%2009-03-03 created by DaiXM
%solidLine =3 for 3 groups lines
% h = axes('Color',[.9 .9 .9],...
%       'GridLineStyle','--',...
%       'ZTickLabels','-1|Z = 0 Plane|+1',...
%       'FontName','times',...
%       'FontAngle','italic',...
%       'FontSize',14,...
%       'XColor',[0 0 .7],...
%       'YColor',[0 0 .7],...
%       'ZColor',[0 0 .7]);
%Color  SymbolLine  Type
% b     blue    .     point    -     solid
% g     green   o     circle  :     dotte
% r     red     x     x-mark    -.    dashdot
% c     cyan    +     plus     --    dashed
% m     magenta *      star
% y     yellows      square
% k     blackd       diamond

% v     triangle (down)
% ^     triangle (up)
% <     triangle (left)
% >     triangle (right)
% p     pentagram
% h     hexagram
Columns=size(DataMatrix,2);
%LineStyles='-ks --r> --bo g^- m- c- m*- y-^ r^- k-* r-* b*- g-. k-* r-* k-* r-* g-. k-* r-* k-* r-*';
%LineStyles1='m-* y-^ r+';
%       b     blue          .     point              -     solid
%       g     green         o     circle             :     dotted
%       r     red           x     x-mark             -.    dashdot
%       c     cyan          +     plus               --    dashed
%       m     magenta       *     star             (none)  no line
%       y     yellow        s     square
%       k     black         d     diamond
%                           v     triangle (down)
%                           ^     triangle (up)
%                           <     triangle (left)
%                           >     triangle (right)
%                           p     pentagram
%                           h     hexagram equivalent 

if solidLine==-100
    MarkerFaeColor='w b g r c m y k b g r c m y k b g r c m y k b g r c m y k';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='w  b g r c m y k b g r c m y k b g r c m y k b g r c m y k';
    MarkerFaeColor='w b g r c m y k b g r c m y k b g r c m y k b g r c m y k';
    ColorStyles='w  b g r c m y k b g r c m y k b g r c m y k b g r c m y k';
    LineStyles='-wo -bo -gv -rd -c< -ms -y> -kh --bo --gv --rd --c< --ms --y> --kh';
%     MarkerFaeColor='b g r c m y k b g r c m y k b g r c m y k b g r c m y k';
    ColorStyles=MarkerFaeColor;
end

if solidLine==0
    MarkerFaeColor='b g r c m y k b g r c m y k b g r c m y k b g r c m y k';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b g r c m y k b g r c m y k b g r c m y k b g r c m y k';
    LineStyles='-bo -gd -rv -c< -ms -y> -kh --bo --gv --rd --c< --ms --y> --kh';   
    
     MarkerFaeColor='g b r c m y k b g r c m y k b g r c m y k b g r c m y k';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='g b r c m y k b g r c m y k b g r c m y k b g r c m y k';
    LineStyles='-gd -bo -rv -c< -ms -y> -kh --bo --gv --rd --c< --ms --y> --kh';    
end

if solidLine==10
    MarkerFaeColor='b r g c r c m y k b g r c m y k b g r c m y k b g r c m y k';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b r g c r c m y k b g r c m y k b g r c m y k b g r c m y k';
    MarkerFaeColor='b r g c r c m y k b g r c m y k b g r c m y k b g r c m y k';
    ColorStyles='b r g c r c m y k b g r c m y k b g r c m y k b g r c m y k';
    LineStyles='-bo -rv -gd -c< -ms -y> -kh --bo --gv --rd --c< --ms --y> --kh';
    MarkerFaeColor='b r g c r c m y k b g r c m y k b g r c m y k b g r c m y k';
    ColorStyles='b r g c r c m y k b g r c m y k b g r c m y k b g r c m y k';
end

if solidLine==2441
    MarkerFaeColor='b g r c b g r c k';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles=MarkerFaeColor;
    MarkerFaeColor=MarkerFaeColor;
    ColorStyles=ColorStyles;
    LineStyles='-bo -gv -rd -c< --bo --gv --rd --c< -kp';
end


if solidLine==2
    MarkerFaeColor='b g b g b g b g b g b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b g b g b g b g b g b g b g b g';
    MarkerFaeColor='b g b g b g b g b g b g b g b g';
    ColorStyles='b g b g b g b g b g b g b g b g';
    LineStyles='-bo -gv --bo --gv -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';
    MarkerFaeColor='b g b g b g b g b g b g b g b g';
    ColorStyles=MarkerFaeColor;
end
if solidLine==241
    MarkerFaeColor='b g b g b g b g r g b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);
    MarkerFaeColor='b g b g b g b g r g b g b g b g';
    MarkerFaeColor='b g b g b g b g r g b g b g b g';
    MarkerFaeColor='b g b g b g b g r g b g b g b g';
    LineStyles='-bo -gv -b> -g< -b^ -gp -b* -gd -ro -g< ';
    MarkerFaeColor='b g b g b g b g r g b g b g b g';
    ColorStyles=MarkerFaeColor;
end
if solidLine==241
    MarkerFaeColor='b g b g b g b g r g b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);
    MarkerFaeColor='b g b g b g b g r g b g b g b g';
    MarkerFaeColor='b g b g b g b g r g b g b g b g';
    MarkerFaeColor='b g b g b g b g r g b g b g b g';
    LineStyles='-bo -.go -bs -.gs -bp -.gp -bh -.gh -ro -g< ';
    MarkerFaeColor='b g b g b g b g r g b g b g b g';
    ColorStyles=MarkerFaeColor;
end
%       b     blue          .     point              -     solid
%       g     green         o     circle             :     dotted
%       r     red           x     x-mark             -.    dashdot
%       c     cyan          +     plus               --    dashed
%       m     magenta       *     star             (none)  no line
%       y     yellow             square
%       k     black         d     diamond
%                           v     triangle (down)
%                           ^     triangle (up)
%                           <     triangle (left)
%                           >     triangle (right)
%                           p     pentagram
%                           h     hexagram equivalent 
if solidLine==42
    MarkerFaeColor='b b b b g g g g g b g b g r g b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);
    MarkerFaeColor='b b b b g g g g g b g b g r g b g b g b g';
    MarkerFaeColor='b b b b g g g g g b g b g r g b g b g b g';
    MarkerFaeColor='b b b b g g g g g b g b g r g b g b g b g';
    LineStyles='-bo -bd -b> -bp -.go -.gd -.g> -.gp -ro -g< ';
    MarkerFaeColor='b b b b g g g g g b g b g r g b g b g b g';
    ColorStyles=MarkerFaeColor;
end
if solidLine==52
    MarkerFaeColor='b b b b b g g g g g g b g b g r g b g b g b g';
   MarkerFaeColor='b b b b b g g g g g g b g b g r g b g b g b g';
   MarkerFaeColor='b b b b b g g g g g g b g b g r g b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);    
    LineStyles='-bo -b> -bh -bp -bd -.go -.g> -.gh -.gp -.gd -ro -g< ';
    ColorStyles=MarkerFaeColor;
end
if solidLine==62
    MarkerFaeColor='b b b b b b g g g g g g g b g b g r g b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);    
    LineStyles='-bo -b> -bv -bp -bd -b< -.go -.g> -.gv -.gp -.gd -.g< -ro -g< ';
    ColorStyles=MarkerFaeColor;
end
if solidLine==67
    MarkerFaeColor='b b b b b b g g g g g g g g b g b g r g b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);    
    LineStyles='-bo -b> -bv -bp -bd -b< -.go -.g> -.gv -.gp -.gd -.g< -.gh -ro -g< ';
    ColorStyles=MarkerFaeColor;
end


if solidLine==72
    MarkerFaeColor='b b b b b b b g g g g g g g g b g b g r g b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);    
    LineStyles='-bo -b> -bv -bh -bp -bd -b< -.go -.g> -.gv -.gh -.gp -.gd -.g< -ro -g< ';
    ColorStyles=MarkerFaeColor;
end


if solidLine==242
    MarkerFaeColor='b g b g b g b g r g b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);
    MarkerFaeColor='b g b g b g b g r g b g b g b g';
    MarkerFaeColor='b g b g b g b g r g b g b g b g';
    MarkerFaeColor='b g b g b g b g r g b g b g b g';
    LineStyles='-bo --go -b> --g> -bs --gs -bd --gd -ro -g< ';
    MarkerFaeColor='b g b g b g b g r g b g b g b g';
    ColorStyles=MarkerFaeColor;
end

%       b     blue          .     point              -     solid
%       g     green         o     circle             :     dotted
%       r     red           x     x-mark             -.    dashdot
%       c     cyan          +     plus               --    dashed
%       m     magenta       *     star             (none)  no line
%       y     yellow        s     square
%       k     black         d     diamond
%                           v     triangle (down)
%                           ^     triangle (up)
%                           <     triangle (left)
%                           >     triangle (right)
%                           p     pentagram
%                           h     hexagram equivalent 

if solidLine==3
    MarkerFaeColor='b g r b g r b g r b g r b g r';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b g r b g r b g r b g r b g r';
    MarkerFaeColor='b g r b g r b g r b g r b g r';
    ColorStyles='b g r b g r b g r b g r b g r';
    LineStyles='-bo -gv --ro --bo --gv --ro -bo -gv -ro';
    MarkerFaeColor='b g r b g r b g r b g r b g r';
    ColorStyles='b g r b g r b g r b g r b g r';
end
if solidLine==23 
   % LineStyles='-bo --bo -gv -gv -rd --rd --gv -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';   
    
    MarkerFaeColor='b b r r g g g b g b g b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b b r r g g g b g b g b g b g b g';
    MarkerFaeColor='b b r r g g g b g b g b g b g b g';
    ColorStyles='b b r r g g g b g b g b g b g b g';
    LineStyles='-bo --bv -rd --rs -g< --g> --gv -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';  
    MarkerFaeColor='b b r r g g g b g b g b g b g b g';
    ColorStyles='b b r r g g g b g b g b g b g b g';
end

if solidLine==32 
   % LineStyles='-bo --bo -gv -gv -rd --rd --gv -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';   
    
    MarkerFaeColor='b b b g g g r r r b b b r r r g g g ';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b b b g g g r r r b b b r r r g g g ';
    MarkerFaeColor='b b b g g g r r r b b b r r r g g g ';
    ColorStyles='b b b g g g r r r b b b r r r g g g ';
    LineStyles='--bo -.bo -bo --gs -.gs -gs --rd -.rd -rd --bo -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';    MarkerFaeColor='b g r b g r b g b g b g b g b g';
    ColorStyles='b b b g g g r r r b b b r r r g g g ';
    
    
    
end
if solidLine==32 
   % LineStyles='-bo --bo -gv -gv -rd --rd --gv -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';   
    
    MarkerFaeColor='b b b g g g r r r b b b r r r g g g ';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b b b g g g r r r b b b r r r g g g ';
    MarkerFaeColor='b b b g g g r r r b b b r r r g g g ';
    ColorStyles='b b b g g g r r r b b b r r r g g g ';
    LineStyles='-bo -bs -bd -gv -g^ -g< -rp -r> -r* --bo -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';    MarkerFaeColor='b g r b g r b g b g b g b g b g';
    ColorStyles='b b b g g g r r r b b b r r r g g g ';
   
    MarkerFaeColor='b b b g g g r r r b b b r r r g g g ';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b b b g g g r r r b b b r r r g g g ';
    MarkerFaeColor='b b b g g g r r r b b b r r r g g g ';
    ColorStyles='b b b g g g r r r b b b r r r g g g ';
    LineStyles='-bo -bs -bd --go --gs --gd -rp -r> -r* --bo -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';    MarkerFaeColor='b g r b g r b g b g b g b g b g';
    ColorStyles='b b b g g g r r r b b b r r r g g g ';

    
    
    
end

%       b     blue          .     point              -     solid
%       g     green         o     circle             :     dotted
%       r     red           x     x-mark             -.    dashdot
%       c     cyan          +     plus               --    dashed
%       m     magenta       *     star             (none)  no line
%       y     yellow        s     square
%       k     black         d     diamond
%                           v     triangle (down)
%                           ^     triangle (up)
%                           <     triangle (left)
%                           >     triangle (right)
%                           p     pentagram
%                           h     hexagram equivalent 



if solidLine==33 
   % LineStyles='-bo --bo -gv -gv -rd --rd --gv -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';   
    
    MarkerFaeColor='b b b r r r g g g b b b r r r g g g ';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b b b r r r g g g b b b r r r g g g ';
    MarkerFaeColor='b b b r r r g g g b b b r r r g g g ';
    ColorStyles='b b b r r r g g g b b b r r r g g g ';
    LineStyles='-bo --bv :bd -ro --rv :rd -go --gv :gd -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';    MarkerFaeColor='b g r b g r b g b g b g b g b g';
    ColorStyles='b b b r r r g g g b b b r r r g g g ';
end
if solidLine==33 
   % LineStyles='-bo --bo -gv -gv -rd --rd --gv -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';   
    
    MarkerFaeColor='b b b r r r g g g b b b r r r g g g ';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b b b r r r g g g b b b r r r g g g ';
    MarkerFaeColor='b b b r r r g g g b b b r r r g g g ';
    ColorStyles='b b b r r r g g g b b b r r r g g g ';
    LineStyles='-bo --bv :bd -ro --rv :rd -go --gv :gd -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';    MarkerFaeColor='b g r b g r b g b g b g b g b g';
    ColorStyles='b b b r r r g g g b b b r r r g g g ';
end
if solidLine==33 
   % LineStyles='-bo --bo -gv -gv -rd --rd --gv -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';   
    
    MarkerFaeColor='b b b r r r g g g b b b r r r g g g ';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b b b r r r g g g b b b r r r g g g ';
    MarkerFaeColor='b b b r r r g g g b b b r r r g g g ';
    ColorStyles='b b b r r r g g g b b b r r r g g g ';
    LineStyles='-bo --bv :bd -ro --rv :rd -go --gv :gd -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';    MarkerFaeColor='b g r b g r b g b g b g b g b g';
    ColorStyles='b b b r r r g g g b b b r r r g g g ';
end


if solidLine==222 
   % LineStyles='-bo --bo -gv -gv -rd --rd --gv -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';   
    
    MarkerFaeColor='b b b r r r g g g b b b r r r g g g ';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b b b r r r g g g b b b r r r g g g ';
    MarkerFaeColor='b b b r r r g g g b b b r r r g g g ';
    ColorStyles='b b b r r r g g g b b b r r r g g g ';
    LineStyles='-bo --bo :bo -rv --rv :rv --gd --gv -gs -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y> -bo -gv -rd -c< -ms -y>';    MarkerFaeColor='b g r b g r b g b g b g b g b g';
    ColorStyles='b b b r r r g g g b b b r r r g g g ';
end


if solidLine==3
    MarkerFaeColor='b g r b g r g b g b g b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b g r b g r b g b g b g b g b g';
    MarkerFaeColor='b g r b g r b g b g b g b g b g';
    ColorStyles='b g r b g r b g b g b g b g b g';
    LineStyles='-bo -gv -rd --bo --gv --rd';
    MarkerFaeColor='b g r b g r b g b g b g b g b g';
    ColorStyles='b g r b g r b g b g b g b g b g';
end

if solidLine==4
    MarkerFaeColor='b g b g b g b g b g b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b g b g b g b g b g b g b g b g';
    MarkerFaeColor='b g b g b g b g b g b g b g b g';
    ColorStyles='b g b g b g b g b g b g b g b g';
    LineStyles='-bo -gv :bo :gv -.bo -.gv --bo --gv -bo -gv';
    MarkerFaeColor='b g b g b g b g b g b g b g b g';
    ColorStyles='b g b g b g b g b g b g b g b g';;
end

if solidLine==44
    MarkerFaeColor='b g r k c b g r k c b g r k c b g r k c b g r k c b g r k c b g r k c b g r k c';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b g r k c b g r k c b g r k c b g r k c b g r k c b g r k c b g r k c b g r k c';
    MarkerFaeColor=ColorStyles;   
    LineStyles='-bo -gv -ro -kv -c> --gv -bo -gv -bo -gv -bo -gv -r> -k< --bo --gv --r> --k<'; 
end
if solidLine==444
    MarkerFaeColor='b g r k b g r k b g r k b g r k b g r k b g r k';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b g r k b g r k b g r k b g r k b g r k b g r k';
    MarkerFaeColor=ColorStyles;   
    LineStyles='-bo -gv -r> -k< --bo --gv --r> --k< -bo -gv -r> -k< --bo --gv --r> --k<'; 
end
if solidLine==344
    MarkerFaeColor='b g b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b g b g b g b g';
    MarkerFaeColor=ColorStyles;   
    LineStyles='-bo -g^ -b> -gv -b> -g< -bs -gd -bp -gh'; 
    
    MarkerFaeColor='b g r b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b g r b g b g g b g';
    MarkerFaeColor=ColorStyles;   
    LineStyles='-bo -g^ -r> -b> -gv -b> -g< -bs -gd -bp -gh';  
    
end

if solidLine==333
    MarkerFaeColor='b g r b g r b g r k b g r k b g r k b g r k';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b g r b g r k b g r k b g r k b g r k b g r k';
    MarkerFaeColor=ColorStyles;   
    LineStyles='-bo -gv -r> --b< --gv --rd --k<';
 %    LineStyles='-bo -gv -r> -k< --bo --gv --r> --k<';
end


if solidLine==24
   MarkerFaeColor='b b g g r r k kb b g g r r k kb b g g r r k kb b g g k k r r';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b b g g r r k kb b g g r r k kb b g g r r k kb b g g k k r r';
    MarkerFaeColor='b b g g r r k kb b g g r r k kb b g g r r k kb b g g k k r r';
    ColorStyles='b b g g r r k kb b g g r r k kb b g g r r k kb b g g k k r r';
    LineStyles='-bo -bv -gd -gs -r> -r< -kp -kh -bo --bo -gv --gv -k> --k> -.ro -.rv -bo --bo -gv --gv -k> --k> -.ro -.rv -bo --bo -gv --gv -k> --k> -.ro -.rv';
    MarkerFaeColor='b b g g r r k kb b g g r r k kb b g g r r k kb b g g k k r r';
    ColorStyles='b b g g r r k kb b g g r r k kb b g g r r k kb b g g k k r r';
end


if solidLine==244
   MarkerFaeColor='b g r k b g r k b g r k b g r k b g r k b g r k';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b g r k b g r k b g r k b g r k b g r k b g r k';
    MarkerFaeColor='b g r k b g r k b g r k b g r k b g r k b g r k';
    ColorStyles='b g r k b g r k b g r k b g r k b g r k b g r k';
    LineStyles='-bo -gv -rd -ks -b> -g< -rp -kh -bo --bo -gv --gv -k> --k> -.ro -.rv -bo --bo -gv --gv -k> --k> -.ro -.rv -bo --bo -gv --gv -k> --k> -.ro -.rv';
    MarkerFaeColor='b g r k b g r k b g r k b g r k b g r k b g r k';
    ColorStyles='b g r k b g r k b g r k b g r k b g r k b g r k';
end



if solidLine==34
   MarkerFaeColor='b b b g g g k k k b b b g g g k k k';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b b b g g g k k k b b b g g g k k k';
    MarkerFaeColor='b b b g g g k k k b b b g g g k k k';
    ColorStyles='b b b g g g k k k b b b g g g k k k';
    LineStyles='-bo --bo  -.bo -gv --gv -.gv -k> --k> -.k> -.gv ';
    MarkerFaeColor=ColorStyles;
    ColorStyles=ColorStyles;
end



%       b     blue          .     point              -     solid
%       g     green         o     circle             :     dotted
%       r     red           x     x-mark             -.    dashdot
%       c     cyan          +     plus               --    dashed
%       m     magenta       *     star             (none)  no line
%       y     yellow        s     square
%       k     black         d     diamond
%                           v     triangle (down)
%                           ^     triangle (up)
%                           <     triangle (left)
%                           >     triangle (right)
%                           p     pentagram
%                           h     hexagram
if solidLine==66
   MarkerFaeColor='b r b r b r b r b r b r b r';
     MarkerFaeColor='b g b g b g b g b g b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles=MarkerFaeColor;
    MarkerFaeColor=MarkerFaeColor;
    ColorStyles=MarkerFaeColor;
    LineStyles='-bo --go -bs --gs -b> --g> -bp --gp -bd --gd -b< --g<';
end
if solidLine==669
    MarkerFaeColor='k k k k k k k k k k k k k k k k k k k k';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles=MarkerFaeColor;
    MarkerFaeColor=MarkerFaeColor;
    ColorStyles=MarkerFaeColor;
    LineStyles='-ko --ko -ks --ks -k> --k> -kp --kp -kd --kd -k< --k<';
end

if solidLine==661
   MarkerFaeColor='b g b g b g b g b g b g b g b g';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles=MarkerFaeColor;
    MarkerFaeColor=MarkerFaeColor;
    ColorStyles=MarkerFaeColor;
    LineStyles='-bo --go -bs --gs -b> --g> -bp --gp -bd --gd -b< --g<';
end

if solidLine==44
   MarkerFaeColor='b g r k b g r k b g r k';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles='b g r k b g r k b g r k';
    MarkerFaeColor='b g r k b g r k b g r k';
    ColorStyles='b g r k b g r k b g r k';
    LineStyles='-bo -gv -ro -kv --bo --gv --ro --kv';
    MarkerFaeColor='b g r k b g r k b g r k';
    ColorStyles='b g r k b g r k b g r k';
end

% if solidLine==33
%    MarkerFaeColor='b g b g b g b g b g b g b g b g';
%     MarkerFaceColor=parse(MarkerFaeColor);
%     ColorStyles='b g b g b g b g b g b g b g b g';
%     MarkerFaeColor='b g b g b g b g b g b g b g b g';
%     ColorStyles='b g b g b g b g b g b g b g b g';
%     LineStyles='-bo -gv --bo --gv :bo :gv -.bo -.gv ';
%     MarkerFaeColor='b g b g b g b g b g b g b g b g';
%     ColorStyles='b g b g b g b g b g b g b g b g';
% end
%

%       b     blue          .     point              -     solid
%       g     green         o     circle             :     dotted
%       r     red           x     x-mark             -.    dashdot
%       c     cyan          +     plus               --    dashed
%       m     magenta       *     star             (none)  no line
%       y     yellow        s     square
%       k     black         d     diamond
%                           v     triangle (down)
%                           ^     triangle (up)
%                           <     triangle (left)
%                           >     triangle (right)
%                           p     pentagram
%                           h     hexagram equivalent 


if solidLine==1
MarkerFaeColor='b b b b b b b b b b b b g g g g g g g g g g g g r r r r r r r r r r r r c c c c c c c c c c c c m m m m m m m m m m m m y y y y y y y y y y y y k k k k k k k k k k k k';
%MarkerFaeColor='b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k b g r c m y k';
MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles=MarkerFaeColor;
    MarkerFaeColor=MarkerFaeColor;
    ColorStyles=MarkerFaeColor;
LineStyles='-bs -bd -bv -b^ -b< -b> -bp -bh -bo -b+ -bx -b* -gs -gd -gv -g^ -g< -g> -gp -gh -go -g+ -gx -g* -rs -rd -rv -r^ -r< -r> -rp -rh -ro -r+ -rx -r* -cs -cd -cv -c^ -c< -c> -cp -ch -co -c+ -cx -c* -ms -md -mv -m^ -m< -m> -mp -mh -mo -m+ -mx -m* -ys -yd -yv -y^ -y< -y> -yp -yh -yo -y+ -yx -y* -ks -kd -kv -k^ -k< -k> -kp -kh -ko -k+ -kx -k*';   
%LineStyles='-bs -bv -rp -co -m< -kd -y> --bs --gv --rp --co --m< --kd --y> -.bs -.gv -.rp -.co -.m< -.kd -.y> -bs -gv -rp -co -m< -kd -y> --bs --gv --rp --co --m< --kd --y> -.bs -.gv -.rp -.co -.m< -.kd -.y>';   
end


% if solidLine==1
%     MarkerFaeColor='b g r c m k b b g r c m k y b g r c m k y b g r c m k y';
%     MarkerFaceColor=parse(MarkerFaeColor);
%     ColorStyles=MarkerFaeColor;
%     MarkerFaeColor=MarkerFaeColor;
%     ColorStyles=MarkerFaeColor;
%     LineStyles='-bs -gv -rp -co -m< -kd -b> --bs --gv --rp --co --m< --kd --y>';
% end
if solidLine==11111
    MarkerFaeColor='g r c m k y b g r c m k y b g r c m k y b g r c m k y';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles=MarkerFaeColor;
    MarkerFaeColor=MarkerFaeColor;
    ColorStyles=MarkerFaeColor;
    LineStyles='-gv -rp -co -m< -kd -y> --bs --gv --rp --co --m< --kd --y>';
end
if solidLine==19
    MarkerFaeColor='k k k k k k k k k k k k k k k k k k k k';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles=MarkerFaeColor;
    MarkerFaeColor=MarkerFaeColor;
    ColorStyles=MarkerFaeColor;
    LineStyles='-ks -kv -kp -ko -k< -kd -k> -ks --gv --rp --co --m< --kd --y>';
end
 
if solidLine==333
    
%     MarkerFaeColor='b b b b b b g g g g g g';
%     MarkerFaeColor='b b b b b b b b b b b b'; 
    MarkerFaeColor='b b b g g g r r r k k k b b b g g g r r r k k k';
    MarkerFaceColor=parse(MarkerFaeColor);
    ColorStyles=MarkerFaeColor;
    MarkerFaeColor=ColorStyles;   
    LineStyles='-bh -bo -bx -g+ -g* -gs -rd -rv -r^ -k< -k> -kp';
     LineStyles=':bd :bv :b^ -.bd -.bv -.b^ -bd -bv -b^ --bd --bv --b^';     
      LineStyles=':b< :bv :b^ -.g< -.gv -.g^ -rh -rv -r^ --rh --rv --r^';
      LineStyles='-b> -bp -bh -.g> -.gp -.gh -r< -rv -r^ -.k< -.kv -.k^ -.k< -.kv -.k^';           
end
%                           v     triangle (down)
%                           ^     triangle (up)
%                           <     triangle (left)
%                           >     triangle (right)
%                           p     pentagram
%                           h     hexagram equivalent 


%MarkerFaceColor=parse(MarkerFaeColor);
Rows=size(DataMatrix,1);
LineStyles=parse(LineStyles);
ColorStyles=parse(ColorStyles);
lengendSize=size(LegendString,1);
LegendString=parse(LegendString);
  axes( 'FontSize',axesFontSize);
if SemiLog==0
    for i=1:Rows
        plot(SNR(i,:),DataMatrix(i,:),LineStyles(i,:),'LineWidth',LineWidth, 'MarkerEdgeColor',ColorStyles(i,:),'MarkerFaceColor',MarkerFaceColor(i,:),'MarkerSize',MarkerSize);
        hold on;
    end
        %legend( LegendString,LegendStringSize,0);
        title(TitleString,'FontSize',TitleStringSize);
        xlabel(xlabelString,'FontSize',xlabelSize);
        ylabel(ylabelString,'FontSize',ylabelSize);
    if min(min(DataMatrix))>0
    axis([min(min(SNR))*1 max(max(SNR))*1 min(min(DataMatrix))*1.0 max(max(DataMatrix))*1.0]);
    else
       axis([min(min(SNR))*1 max(max(SNR))*1 min(min(DataMatrix))*1.0 max(max(DataMatrix))*1.0]);
     
    end
end
% if SemiLog==2
%     for i=1:Rows
%         plot(SNR(i,:),DataMatrix(i,:),LineStyles(i,:),'LineWidth',LineWidth, 'MarkerEdgeColor',ColorStyles(i,:),'MarkerFaceColor',MarkerFaceColor(i,:),'MarkerSize',MarkerSize);
%         hold on;
%     end
%     legend( LegendString,LegendStringSize);
%     title(TitleString,'FontSize',TitleStringSize);
%     xlabel(xlabelString,'FontSize',xlabelSize);
%     ylabel(ylabelString,'FontSize',ylabelSize);
%     axis([min(min(SNR))*0.98 max(max(SNR))*1.02 min(min(DataMatrix))*1.0 max(max(DataMatrix))*1.0]);
% end
if SemiLog==1
    for i=1:Rows
        semilogy(SNR(i,:),DataMatrix(i,:),LineStyles(i,:),'LineWidth',LineWidth, 'MarkerEdgeColor',ColorStyles(i,:),'MarkerFaceColor',MarkerFaceColor(i,:),'MarkerSize',MarkerSize);
        hold on;
    end
    %legend( LegendString,LegendStringSize);
    title(TitleString,'FontSize',TitleStringSize);
    xlabel(xlabelString,'FontSize',xlabelSize);
    ylabel(ylabelString,'FontSize',ylabelSize);
    axis([min(min(SNR))*1 max(max(SNR))*1 min(min(DataMatrix))*1.0 max(max(DataMatrix))*1.0]);
end


if SemiLog==2
   
        for i=1:Rows
            plot(SNR,DataMatrix(i,:),LineStyles(i,:),'LineWidth',LineWidth, 'MarkerEdgeColor',ColorStyles(i,:),'MarkerFaceColor',MarkerFaceColor(i,:),'MarkerSize',MarkerSize);
            hold on;
        end
        legend( LegendString,LegendStringSize);
        title(TitleString,'FontSize',TitleStringSize);
        xlabel(xlabelString,'FontSize',xlabelSize);
        ylabel(ylabelString,'FontSize',ylabelSize); 

    axis([min(min(SNR))*1 max(max(SNR))*1 min(min(DataMatrix))*1.0 max(max(DataMatrix))*1.0]);
end
if SemiLog==3
   
        for i=1:Rows
            semilogy(SNR,DataMatrix(i,:),LineStyles(i,:),'LineWidth',LineWidth, 'MarkerEdgeColor',ColorStyles(i,:),'MarkerFaceColor',MarkerFaceColor(i,:),'MarkerSize',MarkerSize);
            hold on;
        end
        legend( LegendString,LegendStringSize);
        title(TitleString,'FontSize',TitleStringSize);
        xlabel(xlabelString,'FontSize',xlabelSize);
        ylabel(ylabelString,'FontSize',ylabelSize); 

    %axis([min(min(SNR))*0.98 max(max(SNR))*1.02 min(min(DataMatrix))*0.999 max(max(DataMatrix))*1.01]);
%     if Bigaxis==0
%     axis([min(min(SNR))*0.98 max(max(SNR))*1.02 min(min(DataMatrix))*0.999 max(max(DataMatrix))*1.01]);
%     end
   % if Bigaxis==1
    axis([min(min(SNR))*1.0 max(max(SNR))*1.0 min(min(DataMatrix))*1.0 max(max(DataMatrix))*1.0]);
   % end

end

MF=1;

function [x] = parse(inStr)

global strLengthMax
sz=size(inStr);
strLen=sz(2);
x=blanks(strLen);
%x=blanks(strLengthMax);2002/5/12 modify
wordCount=1;
last=0;
for i=1:strLen,
  if inStr(i) == ' '
    wordCount = wordCount + 1;
    x(wordCount,:)=blanks(strLen);
    %x(wordCount,:)=blanks(strLengthMax); 2002/5/12 modify
    last=i;
  else
    x(wordCount,i-last)=inStr(i);
  end
end 
