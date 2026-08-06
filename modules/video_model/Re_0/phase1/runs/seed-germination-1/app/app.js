const NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const clamp = (v, a=0, b=1) => Math.max(a, Math.min(b, v));
const mix = (a,b,p) => a + (b-a)*p;
const smooth = (a,b,t) => { const p=clamp((t-a)/(b-a)); return p*p*(3-2*p); };
const segment = (a,b,t) => smooth(a,b,t);
const setOpacity = (id,v) => $(id).setAttribute("opacity", String(clamp(v)));
const setDash = (id,p,length) => { const el=$(id); el.style.strokeDasharray=`${length} ${length}`; el.style.strokeDashoffset=String(length*(1-clamp(p))); };

const captions = [
  [0,3.8,"一颗看似安静的种子，已经装着胚根、胚芽和储存养分的子叶。"],
  [3.8,7.4,"萌发需要水、适宜温度和氧气；三者缺一，过程都会受阻。"],
  [7.4,11.2,"水沿土壤孔隙进入种子，细胞吸水，种子体积明显膨大。"],
  [11.2,15,"适温让酶正常工作；氧气支持呼吸，把储藏物质转成可用能量。"],
  [15,19,"膨胀产生的压力使种皮在胚根端裂开，胚根最先突破种皮。"],
  [19,23.3,"胚根受重力引导向下生长，根冠保护前端，根毛扩大吸收面积。"],
  [23.3,27.2,"主根继续深入，侧根分枝，幼苗开始稳定吸收水和无机盐。"],
  [27.2,31.4,"与此同时，胚轴形成弯钩向上顶土，保护柔嫩的胚芽。"],
  [31.4,35.5,"胚轴穿出地表后逐渐伸直，把两片子叶带到光下。"],
  [35.5,39.6,"子叶展开，继续向幼苗输送储藏养分，也能进行少量光合作用。"],
  [39.6,43.6,"顶部生长点继续分裂；真正的叶片在两片子叶之间出现。"],
  [43.6,48.2,"第一片真叶展开，叶脉和叶面积增大，光合作用成为主要能量来源。"],
  [48.2,52,"从吸水到真叶出现，种子完成萌发，并建立起独立生长的幼苗。"]
];

const stageData = [
  [0,7.4,"1","萌发条件"],[7.4,14.8,"2","吸水与启动"],[14.8,23.2,"3","胚根突破"],
  [23.2,31.4,"4","根深芽高"],[31.4,39.6,"5","子叶展开"],[39.6,52.1,"6","真叶出现"]
];

function makeDrops(){
  const g=$("waterDrops");
  const xs=[770,835,900,982,1045,1115,1190,1280,690,1360,875,1240];
  xs.forEach((x,i)=>{
    const path=document.createElementNS(NS,"path");
    path.setAttribute("d","M0-15C-10 0-15 8-15 17A15 15 0 0015 17C15 8 10 0 0-15Z");
    path.setAttribute("transform",`translate(${x} ${360+(i%4)*58}) scale(${.65+(i%3)*.12})`);
    path.dataset.x=x; path.dataset.base=350+(i%4)*46; path.dataset.speed=38+(i%5)*9;
    g.appendChild(path);
  });
}
function makeOxygen(){
  const g=$("oxygenBubbles");
  [[780,482],[855,650],[1110,526],[1212,705],[705,780],[1320,590]].forEach(([x,y],i)=>{
    const group=document.createElementNS(NS,"g");
    group.innerHTML=`<circle cx="${x}" cy="${y}" r="24" fill="#78a8e8" opacity=".72" stroke="#cce1ff" stroke-width="3"/><text x="${x}" y="${y+7}" text-anchor="middle" font-size="17" font-weight="900" fill="white">O₂</text>`;
    group.dataset.y=y; group.dataset.i=i; g.appendChild(group);
  });
}
makeDrops(); makeOxygen();

function updateCallout(t,title,body,x=1280,y=415,line="M1284 495C1197 490 1100 520 1030 558"){
  const visible=title?1:0; setOpacity("callout",visible);
  if(!visible) return;
  $("calloutTitle").textContent=title; $("calloutBody").textContent=body;
  $("calloutBox").setAttribute("x",x); $("calloutBox").setAttribute("y",y);
  $("calloutTitle").setAttribute("x",x+40); $("calloutTitle").setAttribute("y",y+50);
  $("calloutBody").setAttribute("x",x+40); $("calloutBody").setAttribute("y",y+90);
  $("calloutLine").setAttribute("d",line);
}

function renderState(t){
  const waterP=segment(4.5,10.5,t);
  const swellP=segment(6.2,10.3,t);
  const crackP=segment(10.8,14.0,t);
  const rootP=segment(15.0,22.0,t);
  const branchP=segment(21.8,27.0,t);
  const shootP=segment(24.0,33.5,t);
  const cotP=segment(34.8,38.0,t);
  const leafP=segment(40.8,47.0,t);

  // Environmental movement remains deterministic at every absolute time.
  setOpacity("waterDrops", clamp(segment(3.8,5.2,t)*(1-segment(12.0,14.0,t))));
  [...$("waterDrops").children].forEach((el,i)=>{
    const phase=Math.max(0,t-4)*Number(el.dataset.speed)+i*31;
    const y=Number(el.dataset.base)+(phase%310);
    el.setAttribute("transform",`translate(${el.dataset.x} ${y}) scale(${.65+(i%3)*.12})`);
  });
  setOpacity("oxygenBubbles", clamp(segment(5.0,7.0,t)*(1-segment(14,16,t))));
  [...$("oxygenBubbles").children].forEach((el)=>{
    const y=Number(el.dataset.y)+Math.sin(t*1.3+Number(el.dataset.i))*10;
    el.setAttribute("transform",`translate(0 ${y-Number(el.dataset.y)})`);
  });
  setOpacity("metabolicGlow", segment(8.0,10.5,t)*(1-segment(16,18,t)));

  const scale=mix(0.82,1.06,swellP);
  const splitShift=mix(0,14,crackP);
  $("seed").setAttribute("transform",`translate(960 570) rotate(-8) scale(${scale}) translate(${-splitShift*.1} 0)`);
  const coatFade=segment(33.5,39.0,t);
  $("seedCoat").setAttribute("opacity",String((1-.52*crackP)*(1-.68*coatFade)));
  $("cotyledonCore").setAttribute("opacity",String(crackP*.98*(1-.92*cotP)));
  setOpacity("coatCrack",crackP); setOpacity("coatFlaps",crackP*(1-.82*cotP));
  $("hilum").setAttribute("opacity",String(.8*(1-crackP*.6)));

  setDash("mainRoot",rootP,500);
  ["sideRoot1","sideRoot2","sideRoot3","sideRoot4"].forEach((id,i)=>setDash(id,clamp(branchP-i*.12),250));
  setOpacity("rootHairs",segment(21.0,23.0,t));
  setOpacity("rootCap",segment(16.0,17.0,t));
  const rootPath=$("mainRoot");
  const rootLength=rootPath.getTotalLength();
  const rootPoint=rootPath.getPointAtLength(Math.max(1,rootLength*rootP));
  $("rootCap").setAttribute("transform",`translate(${rootPoint.x-930} ${rootPoint.y-973})`);

  setDash("stem",shootP,560); setDash("stemHighlight",shootP,560);
  setOpacity("closedBud",segment(31.2,33.4,t)*(1-cotP));
  setOpacity("cotyledonLeaves",cotP);
  const cotScale=.35+.65*cotP;
  $("cotyledonLeaves").setAttribute("transform",`translate(${1064*(1-cotScale)} ${242*(1-cotScale)}) scale(${cotScale})`);
  setOpacity("trueLeaves",leafP);
  const leafScale=.18+.82*leafP;
  $("trueLeaves").setAttribute("transform",`translate(${1065*(1-leafScale)} ${188*(1-leafScale)}) scale(${leafScale})`);
  setOpacity("sunHalo",.24+.22*segment(30,42,t));

  // Teaching annotations.
  const condShow=1-segment(14.5,16.5,t);
  setOpacity("conditionPanel",condShow);
  $("conditionWater").setAttribute("opacity",String(t>=3.8?1:.62));
  $("conditionTemp").setAttribute("opacity",String(t>=5?1:.62));
  $("conditionOxygen").setAttribute("opacity",String(t>=6.2?1:.62));
  if(t>=7.4&&t<11.2) updateCallout(t,"吸水膨胀","体积增大，内部代谢重新启动");
  else if(t>=11.2&&t<15) updateCallout(t,"呼吸供能","储藏物质 → 可用能量（ATP）",1280,415,"M1284 495C1200 492 1114 520 1030 558");
  else if(t>=15&&t<19) updateCallout(t,"种皮破裂","胚根端先形成清楚裂口",1280,448,"M1284 518C1180 540 1092 566 1045 610");
  else if(t>=19&&t<23.3) updateCallout(t,"根冠 + 根毛","保护生长点，扩大吸收面积",1250,650,"M1254 720C1135 748 1038 852 947 958");
  else if(t>=27.2&&t<35.5) updateCallout(t,"胚轴弯钩","顶土时保护柔嫩的胚芽",1290,360,"M1294 438C1193 429 1100 423 1023 439");
  else if(t>=35.5&&t<39.6) updateCallout(t,"子叶","短期供应储藏养分",1350,337,"M1355 409C1265 359 1170 295 1092 259");
  else if(t>=43.6) updateCallout(t,"真叶","成为主要光合器官",1370,292,"M1374 363C1307 296 1245 220 1171 168");
  else updateCallout(t,null,null);
  setOpacity("directionArrows",segment(18,20,t)*(1-segment(35.5,37,t)));
  setOpacity("organLabels",segment(23,24,t));
  $("rootLabel").setAttribute("opacity",String(segment(23,25,t)*(1-segment(30,32,t))));
  $("cotLabel").setAttribute("opacity",String(cotP*(1-segment(42.5,44.5,t))));
  $("leafLabel").setAttribute("opacity",String(leafP));

  const stage=stageData.find(([a,b])=>t>=a&&t<b) || stageData.at(-1);
  $("stageNumber").textContent=stage[2]; $("stageText").textContent=stage[3];
  $("progressFill").setAttribute("width",String(890*clamp(t/52)));
  const cue=captions.find(([a,b])=>t>=a&&t<b);
  $("subtitleText").textContent=cue?cue[2]:"";
}

window.LIVE_DOCUMENT_META={duration:52,fps:24,width:1920,height:1080};
window.LIVE_DOCUMENT_BRIDGE={
  version:1,
  route:"realizable",
  targetStyle:"scientific_realism",
  reason:"同一颗豆类种子在固定土壤剖面中连续萌发，主体的形态、方向和空间拓扑清楚，可在不改变教学结构的前提下转换为科学真实感；教学标注与字幕已独立分层。",
  worldContinuity:[
    "固定的侧视土壤剖面、地表高度与柔和晨光方向保持不变",
    "同一颗棕褐色双子叶种子始终位于画面中央，其胚根端朝右下方",
    "主根由种子向下延伸并形成侧根，绿色胚轴由同一胚体向上穿出地表",
    "子叶先于第一片真叶展开；真叶位于两片子叶上方并保留清楚叶脉"
  ],
  posterMomentId:"true_leaf_post",
  keyMoments:[
    {id:"coat_pre",time:10.4,kind:"pre_event",description:"种子完成吸胀但种皮仍完整，是破裂前的稳定状态。",eventId:"coat_rupture",visibleObjects:["hydrated_seed","intact_seed_coat","soil_profile"],preserve:["种子位于地表下中央","种皮完整且胚根端朝右下","种子较初始状态明显膨大"],realizable:true},
    {id:"coat_post",time:14.4,kind:"post_event",description:"种皮在胚根端形成稳定裂口，内部子叶可见。",eventId:"coat_rupture",visibleObjects:["split_seed_coat","cotyledons","soil_profile"],preserve:["裂口只位于胚根端","两片子叶仍在种皮内","镜头与土壤位置不变"],realizable:true},
    {id:"radicle_pre",time:14.7,kind:"pre_event",description:"种皮已经裂开，但胚根尚未形成可见长度。",eventId:"radicle_appearance",visibleObjects:["split_seed_coat","embryo_root_tip"],preserve:["裂口与胚根尖端相连","主根尚未向深层土壤延伸"],realizable:true},
    {id:"radicle_post",time:22.2,kind:"post_event",description:"胚根已稳定向下延伸，并出现根冠和根毛。",eventId:"radicle_appearance",visibleObjects:["main_root","root_cap","root_hairs","seed"],preserve:["主根由种子裂口连续向下","根冠位于最下端","根毛分布在较成熟根段"],realizable:true},
    {id:"shoot_pre",time:23.4,kind:"pre_event",description:"地下根系已建立，胚芽尚未穿出地表。",eventId:"shoot_emergence",visibleObjects:["seed","main_root","side_root_buds","soil_surface"],preserve:["地表完整","根系和种子保持连接","胚芽位于地下"],realizable:true},
    {id:"shoot_post",time:34.0,kind:"post_event",description:"弯钩状胚轴穿出地表并开始伸直。",eventId:"shoot_emergence",visibleObjects:["hypocotyl_hook","soil_surface","root_system","cotyledon_bud"],preserve:["胚轴从原种子连续向上","弯钩顶端在地表之上","根系保持向下"],realizable:true},
    {id:"cotyledon_pre",time:34.4,kind:"pre_event",description:"胚轴已经出土，子叶仍合拢在顶部。",eventId:"cotyledon_opening",visibleObjects:["upright_hypocotyl","closed_cotyledons","root_system"],preserve:["合拢子叶位于胚轴顶端","茎根轴线连续"],realizable:true},
    {id:"cotyledon_post",time:38.4,kind:"post_event",description:"两片子叶在地表上方完整展开。",eventId:"cotyledon_opening",visibleObjects:["open_cotyledons","hypocotyl","root_system"],preserve:["两片子叶左右展开","子叶低于后续真叶位置","根茎仍与原种子相连"],realizable:true},
    {id:"true_leaf_pre",time:40.4,kind:"pre_event",description:"子叶稳定展开，中央生长点尚无完整真叶。",eventId:"true_leaf_appearance",visibleObjects:["open_cotyledons","apical_bud","root_system"],preserve:["中央生长点位于两片子叶之间","尚无展开真叶"],realizable:true},
    {id:"true_leaf_post",time:47.4,kind:"post_event",description:"第一片真叶在子叶上方稳定展开，幼苗结构完整。",eventId:"true_leaf_appearance",visibleObjects:["first_true_leaves","open_cotyledons","stem","branched_root_system"],preserve:["真叶位于子叶上方","真叶有清楚叶脉且左右展开","地上与地下器官由同一轴线相连"],realizable:true}
  ],
  events:[
    {id:"coat_rupture",type:"topology_change",objects:["seed_coat","cotyledons"],preMomentId:"coat_pre",postMomentId:"coat_post"},
    {id:"radicle_appearance",type:"object_appearance",objects:["main_root","root_cap","root_hairs"],preMomentId:"radicle_pre",postMomentId:"radicle_post"},
    {id:"shoot_emergence",type:"topology_change",objects:["hypocotyl_hook","soil_surface"],preMomentId:"shoot_pre",postMomentId:"shoot_post"},
    {id:"cotyledon_opening",type:"topology_change",objects:["left_cotyledon","right_cotyledon"],preMomentId:"cotyledon_pre",postMomentId:"cotyledon_post"},
    {id:"true_leaf_appearance",type:"object_appearance",objects:["first_true_leaves"],preMomentId:"true_leaf_pre",postMomentId:"true_leaf_post"}
  ]
};

window.renderFrame=async function(t,options={}){
  t=clamp(Number(t)||0,0,52);
  const mode=options.mode??"presentation";
  const world=$("world"), overlay=$("overlay"), subtitles=$("subtitles");
  document.body.style.background=mode==="overlay"?"transparent":"#c9eff3";
  world.style.display=mode==="overlay"?"none":"";
  overlay.style.display=mode==="clean"?"none":"";
  subtitles.style.display=mode==="presentation"?"":"none";
  renderState(t);
};
renderState(0);
window.__LIVE_DOCUMENT_READY__=true;
