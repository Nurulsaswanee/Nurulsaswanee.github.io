// dataController.js
let students=[]; //กำหนดค่าเริ่มต้นให้กับอาร์เรย์ของนักศึกษา

function getAllStudents(){
    return students;
}
function getStudentByID(std_id){
    for(let i=0; i<students.length; i++){
        if(students[i].stid == std_id){
            return students[i];
        }
    }
    return -1; //เพื่อระบุว่าไม่มีข้อมูลนักศึกษาที่มีรหัส = std_id ที่ส่งมา
}
//ฟังก์ชันสำหรับเพิ่มนักศึกษาใหม่ new_std เข้าไปในรายการข้อมูล students
function addStudent(new_std){
    students.push(new_std); 
}
//
function delStudentByID(std_id){
    //สร้าง students ใหม่โดยกรองเอาเฉพาะที่มีไอดีไม่เท่ากับ std_id
    students = students.filter(student => student.stid != std_id);
}
function delStudentByName(nme) {
    //สร้าง students ใหม่โดยกรองเอาเฉพาะที่มีไอดีไม่เท่ากับ std_id
    students = students.filter(student => student.name != nme);
}
function updateStudent(studentObj) {
    let index = students.findIndex(s => s.stid === studentObj.stid);
    if (index !== -1) {
        students[index] = studentObj;
    }
}